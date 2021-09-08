import sys
import gin
import time

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import logging
import subprocess
import argparse
import numpy as np

# import c2nl.config as config
import c2nl.inputters.utils as util
from c2nl.inputters import constants

from collections import OrderedDict, Counter
# from tqdm import tqdm
from c2nl.inputters.timer import AverageMeter, Timer
import c2nl.inputters.vector as vector
import c2nl.inputters.dataset as data

from c2nl.model import Code2NaturalLanguage
from c2nl.eval.bleu import corpus_bleu
# from c2nl.eval.rouge import Rouge
# from c2nl.eval.meteor import Meteor
from util.LoggerUtil import info_logger, set_logger, torch_summarize, count_parameters
from util.EvaluateUtil import bleu_so_far, metetor_rouge_cider
from util.DataUtil import get_object_attr, make_directory, read_pickle_data, get_config_str, time_format
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

# logger = logging.getLogger()
if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable
from util.Config import Config as cf
import warnings

warnings.filterwarnings('ignore')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    src_dict = util.build_word_and_char_dict(args,
                                             examples=train_exs + dev_exs,
                                             fields=['code'],
                                             dict_size=args.code_vocab_size,
                                             no_special_token=True)
    ast_word_count = read_pickle_data(args.ast_original_token_wc)
    if args.use_asts:
        ast_dict = util.build_word_and_char_dict(args,
                                                 examples=None,
                                                 fields=['ast'],
                                                 dict_size=args.ast_vocab_size,
                                                 no_special_token=True,
                                                 words=list(ast_word_count.keys()))

    else:
        ast_dict = None

    tgt_dict = util.build_word_and_char_dict(args,
                                             examples=train_exs + dev_exs,
                                             fields=['summary'],
                                             dict_size=args.summary_vocab_size,
                                             no_special_token=False)

    logger.info('Num words in source = %d and target = %d' % (len(src_dict), len(tgt_dict)))

    # Initialize model
    # model = Code2NaturalLanguage(config.get_model_args(args), src_dict, tgt_dict)
    model = Code2NaturalLanguage(args, src_dict, tgt_dict, state_dict=None, ast_dict=ast_dict)
    # model.use_cuda = True
    if args.initialize_weights:
        initialize_weights(model.network)
    return model


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    perplexity = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    pbar = tqdm(data_loader)
    #
    # pbar.set_description("%s" % 'Epoch = %d [perplexity = x.xx, ml_loss = x.xx]' %
    #                      current_epoch)
    if sys.stderr.isatty():
        pbar.set_description("%s" % 'Epoch = %d [perplexity = x.xx, ml_loss = x.xx]' %
                             current_epoch)
    # Run one epoch
    for idx, ex in enumerate(pbar):
        # try:
        bsz = ex['batch_size']
        if args.optimizer in ['sgd', 'adam'] and current_epoch <= args.warmup_epochs:
            cur_lrate = global_stats['warmup_factor'] * (model.updates + 1)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lrate

        net_loss = model.update(ex)
        ml_loss.update(net_loss['ml_loss'], bsz)
        perplexity.update(net_loss['perplexity'], bsz)
        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.2f]' % \
                   (current_epoch, perplexity.avg, ml_loss.avg)
        if sys.stderr.isatty():
            pbar.set_description("%s" % log_info)
        # except RuntimeError:
        #     logger.info("idx error%s" % str(ex["ids"]))
        # break
    logger.info('train: Epoch %d | perplexity = %.2f | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        # model.checkpoint(args.model_file + '.checkpoint', current_epoch + 1)
        if args.local_rank == 0:
            model.checkpoint(args.model_save_path + '.checkpoint', current_epoch + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, mode='dev'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for idx, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            # ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
            ex_ids = ex["ids"]
            predictions, targets, attn_info = model.predict(ex, replace_unk=True)
            # predictions, targets, copy_info = model.predict(ex, replace_unk=False)

            src_sequences = [code for code in ex['code_text']]
            examples += batch_size
            for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                hypotheses[key] = [pred]
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                sources[key] = src
            copy_info = attn_info['copy_info']
            if copy_info is not None:
                copy_info = copy_info.cpu().numpy().astype(int).tolist()
                for key, cp in zip(ex_ids, copy_info):
                    copy_dict[key] = cp
            if sys.stderr.isatty():
                pbar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])
    preds, refs, refs_truncation = zip(
        *[(hypotheses[fid][0].split(), [references[fid][0].split()], [references[fid][0].split()[:args.sum_max_len]])
          for fid in hypotheses.keys()])
    ret_val_full, _, bleu_val_full = bleu_so_far(refs, preds)
    result = dict()
    result['bleu'] = bleu_val_full
    logger.info('valid valid official: Epoch = %d | ' %
                (global_stats['epoch']) +
                'bleu = %.2f ' % bleu_val_full)
    if args.only_test:
        copy_dict = None if len(copy_dict) == 0 else copy_dict
        bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
                                                                       references,
                                                                       copy_dict,
                                                                       sources=sources,
                                                                       filename=args.pred_file,
                                                                       print_copy_info=args.print_copy_info,
                                                                       mode=mode)
        result = dict()
        result['bleu'] = bleu
        result['rouge_l'] = rouge_l
        result['meteor'] = meteor
        result['precision'] = precision
        result['recall'] = recall
        result['f1'] = f1
        score_Rouge, score_Cider, score_Meteor = metetor_rouge_cider(refs, preds)
        logger.info('Rouge= %.2f | Cider = %.2f | Meteor = %.2f | ' % (score_Rouge, score_Cider, score_Meteor))
    #     if mode == 'test':
    #         logger.info('test valid official: '
    #                     'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
    #                     (bleu, rouge_l, meteor) +
    #                     'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
    #                     'examples = %d | ' %
    #                     (precision, recall, f1, examples) +
    #                     'test time = %.2f (s)' % eval_time.time())
    #
    #     else:
    #         logger.info('dev valid official: Epoch = %d | ' %
    #                     (global_stats['epoch']) +
    #                     'bleu = %.2f | rouge_l = %.2f | '
    #                     'Precision = %.2f | Recall = %.2f | F1 = %.2f | examples = %d | ' %
    #                     (bleu, rouge_l, precision, recall, f1, examples) +
    #                     'valid time = %.2f (s)' % eval_time.time())
    # else:

    # ret_val_truncation, _, bleu_val_truncation = bleu_so_far(refs_truncation, preds)
    # logger.info("---------use_full_sum = False: ------\n %s" % ret_val_truncation)
    # if "tl_codesum" in args.dataset_type and args.rm_duplication:
    #     duplication_fid_of_test_in_tl_codesum = read_pickle_data(args.duplication_fid_of_test_in_tl_codesum_path)
    #     preds, refs, refs_truncation = zip(
    #         *[(
    #             hypotheses[fid][0].split(), [references[fid][0].split()],
    #             [references[fid][0].split()[:args.sum_max_len]])
    #             for fid in hypotheses.keys() if fid not in duplication_fid_of_test_in_tl_codesum])
    #     ret_val_truncation, _, bleu_val_truncation = bleu_so_far(refs_truncation, preds)
    #     logger.info("---------use_full_sum = False (rm_duplication): ------\n %s" % ret_val_truncation)
    #     ret_val_full, _, bleu_val_full = bleu_so_far(refs, preds)
    #     logger.info("---------use_full_sum = True (rm_duplication): ------\n %s" % ret_val_full)

    return result


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1


def eval_accuracies(hypotheses, references, copy_info, sources=None,
                    filename=None, print_copy_info=False, mode='dev'):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # # Compute ROUGE scores
    # rouge_calculator = Rouge()
    # rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)
    #
    # if mode == 'test':
    #     meteor = 0
    #     # meteor_calculator = Meteor()
    #     # meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    # else:
    #     meteor = 0
    meteor, rouge_l = 0, 0
    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0],
                                              references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        if fw:
            if copy_info is not None and print_copy_info:
                prediction = hypotheses[key][0].split()
                pred_i = [word + ' [' + str(copy_info[key][j]) + ']'
                          for j, word in enumerate(prediction)]
                pred_i = [' '.join(pred_i)]
            else:
                pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            logobj['references'] = references[key][0] if cf.print_one_target \
                else references[key]
            logobj['bleu'] = ind_bleu[key]
            # logobj['rouge_l'] = ind_rouge[key]
            fw.write(json.dumps(logobj) + '\n')

    if fw: fw.close()
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(config):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')

    # train_exs = []
    if not config.only_test:
        # config.dataset_weights = dict()
        # for train_src, train_src_tag, train_tgt, dataset_name in \
        #         zip([config.train_src_files], config.train_src_tag_files,
        #             [config.train_tgt_files], [config.dataset_name]):
        train_files = dict()
        train_files['src'] = config.train_src_files
        train_files['ast'] = config.train_ast_files
        if config.is_rebuild_tree:
            train_files['rebuild_tree'] = config.train_rebuild_tree_files
        else:
            train_files['rebuild_tree'] = None
        train_files['src_tag'] = config.train_src_tag_files
        train_files['tgt'] = config.train_tgt_files
        train_exs = util.load_data(config,
                                   train_files,
                                   max_examples=config.max_examples,
                                   dataset_name=config.dataset_name)


        logger.info('Num train examples = %d' % len(train_exs))
    dev_files = dict()
    dev_files['src'] = config.dev_src_files
    dev_files['ast'] = config.dev_ast_files
    dev_files['rebuild_tree'] = config.dev_rebuild_tree_files
    dev_files['src_tag'] = None
    dev_files['tgt'] = config.dev_tgt_files
    dev_exs = util.load_data(config,
                             dev_files,
                             max_examples=config.max_examples,
                             dataset_name=config.dataset_name,
                             test_split=True)
    # dev_exs.extend(exs)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if (not os.path.isfile(config.model_save_path + '.checkpoint')) and (args_cmd.local_rank != 0 and config.is_DDP):
        time.sleep(60)
    if config.only_test:
        if config.pretrained:
            model = Code2NaturalLanguage.load(config.pretrained)
        else:
            # if not os.path.isfile(config.model_file):
            if not os.path.isfile(config.model_save_path):
                raise IOError('No such file: %s' % config.model_save_path)
            model = Code2NaturalLanguage.load(config.model_save_path)
    else:
        # if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        if config.checkpoint and os.path.isfile(config.model_save_path + '.checkpoint'):
            # Just resume training, no modifications.
            logger.info('Found a checkpoint...')
            # checkpoint_file = args.model_file + '.checkpoint'
            checkpoint_file = config.model_save_path + '.checkpoint'
            model, start_epoch = Code2NaturalLanguage.load_checkpoint(checkpoint_file, config.use_gpu)
            if not config.is_DDP:
                model.checkpoint(config.model_save_path + '.checkpoint', start_epoch)
        else:
            if config.pretrained:
                logger.info('Using pretrained model...')
                model = Code2NaturalLanguage.load(config.pretrained, config)
            else:
                logger.info('Training model from scratch...')
                model = init_from_scratch(config, train_exs, dev_exs)

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.info('Trainable #parameters [encoder-decoder] {} [total] {}'.format(
                human_format(model.network.count_encoder_parameters() +
                             model.network.count_decoder_parameters()),
                human_format(model.network.count_parameters())))
            table = model.network.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s' % table)
            model.checkpoint(config.model_save_path + '.checkpoint', 1)
            # model.checkpoint('~/model.checkpoint', 1)

    if config.use_asts:
        if not config.only_test:
            for i, item in enumerate(train_exs):
                value = item["ast"][1]
                util.tree2idx(value, model.ast_dict)
                # train_exs[i]["ast"][1] = value
        for i, item in enumerate(dev_exs):
            value = item["ast"][1]
            util.tree2idx(value, model.ast_dict)
            # dev_exs[i]["ast"][1] = value

    local_rank = 0
    if cf.is_DDP:
        model.parallel = True
        # torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        cf.local_rank = local_rank
        logger.info("local rank %d " % local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.network.to(device)
        if torch.cuda.device_count() > 1:
            logger.info("Let's use %d GPUs!" % torch.cuda.device_count())
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,)
            model.network = torch.nn.parallel.DistributedDataParallel(model.network, device_ids=[device],
                                                                      output_device=local_rank,
                                                                      find_unused_parameters=True)
    else:
        # model.network.to(cf.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.network.to(device)
        model.use_cuda = config.use_gpu
    # Set up optimizer

    # model.init_optimizer()
    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    if not config.only_test:

        train_dataset = data.CommentDataset(train_exs, model)
        if config.sort_by_len:
            train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                    config.batch_size,
                                                    shuffle=True)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
        if cf.is_DDP:
            train_sampler = DistributedSampler(train_sampler)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.data_workers,
            collate_fn=vector.batchify,
            # pin_memory=args.cuda,
            pin_memory=config.use_gpu,
            # drop_last=args.parallel
            drop_last=True
        )

    dev_dataset = data.CommentDataset(dev_exs, model)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    # if cf.is_DDP:
    #     dev_sampler = DistributedSampler(dev_sampler)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=config.test_batch_size,
        sampler=dev_sampler,
        num_workers=config.data_workers,
        collate_fn=vector.batchify,
        pin_memory=config.use_gpu,
        # pin_memory=args.cuda,
        # drop_last=args.parallel
        drop_last=True
        # drop_last=False
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    # logger.info('CONFIG:\n%s' %
    #             json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info('CONFIG:\n%s' % get_object_attr(config))

    # --------------------------------------------------------------------------
    # DO TEST

    if config.only_test:
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
        s_time = time.perf_counter()
        validate_official(config, dev_loader, model, stats, mode='test')
        logger.info("test /epoch time cost: %s" %
                    time_format(time.perf_counter() - s_time))
    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}

        if config.optimizer in ['sgd', 'adam'] and config.warmup_epochs >= start_epoch:
            logger.info("Use warmup rate for the %d epoch, from 0 up to %s." %
                        (config.warmup_epochs, config.lr))
            num_batches = len(train_loader.dataset) // config.batch_size
            warmup_factor = (config.lr + 0.) / (num_batches * config.warmup_epochs)
            stats['warmup_factor'] = warmup_factor

        # for epoch in range(start_epoch, args.num_epochs + 1):
        logger.info(get_config_str())
        for epoch in range(start_epoch, config.epochs + 1):
            s_time = time.perf_counter()
            stats['epoch'] = epoch
            if config.optimizer in ['sgd', 'adam'] and epoch > config.warmup_epochs:
                model.optimizer.param_groups[0]['lr'] = \
                    model.optimizer.param_groups[0]['lr'] * config.lr_decay

            train(config, train_loader, model, stats)
            logger.info("train /epoch time cost: %s" %
                        time_format(time.perf_counter() - s_time))
            s_time = time.perf_counter()
            result = validate_official(config, dev_loader, model, stats)
            logger.info("val /epoch time cost: %s" %
                        time_format(time.perf_counter() - s_time))
            if result[config.valid_metric] > stats['best_valid']:
                logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                            (config.valid_metric, result[config.valid_metric],
                             stats['epoch'], model.updates))
                if local_rank == 0:
                    model.save(config.model_save_path)
                stats['best_valid'] = result[config.valid_metric]
                stats['no_improvement'] = 0
            else:
                stats['no_improvement'] += 1
                if stats['no_improvement'] >= config.early_stop:
                    break
        logger.info("Best epoch: %d", stats['epoch'])
        logger.info(get_config_str())


@gin.configurable
def set_config(arg_idx, code_max_len, sum_max_len, model_type, max_relative_pos, summary_embedding_dim,
               code_embedding_dim,
               dropout_emb, nhead, enc_dropout, use_all_enc_layers, split_decoder, coverage_attn, dec_dropout,
               reload_decoder_state, layer_wise_attn, attn_type, force_copy, lr, sort_by_len, test_batch_size,
               warmup_epochs, epochs, batch_size, print_copy_info, early_stop, checkpoint, aim, clip,
               summary_vocab_size, code_vocab_size, enc_layers, dec_layers, ast_vocab_size=100, RvNN_input_dim=512,
               d_ff=2048, use_asts=True, initialize_weights=False, copy_attn=True,
               node_embedding_dim=100, lr_decay=0.99, use_ast_sub_token=False, aggregate_dim=512,
               dataset_type="codesearchnet_v1", activate_f="tanh"):
    cf.arg_idx = arg_idx
    cf.dataset_type = dataset_type
    if dataset_type == "Funcom":
        cf.dataset_path = "../Data/Funcom"
    elif "TL_CodeSum" in dataset_type:
        cf.dataset_path = "../Data/TL_CodeSum"
    else:
        raise RuntimeError('Unsupported dataset_type: %s' % dataset_type)
    cf.code_max_len = code_max_len
    cf.sum_max_len = sum_max_len
    cf.model_type = model_type
    if "RvNN" in cf.model_type:
        cf.is_rebuild_tree = True
    else:
        cf.is_rebuild_tree = False

    cf.max_relative_pos = max_relative_pos
    cf.summary_embedding_dim = summary_embedding_dim
    cf.code_embedding_dim = code_embedding_dim
    cf.node_embedding_dim = node_embedding_dim
    cf.dropout_emb = dropout_emb

    cf.nhead = nhead
    cf.enc_dropout = enc_dropout
    cf.use_all_enc_layers = use_all_enc_layers

    cf.split_decoder = split_decoder
    cf.coverage_attn = coverage_attn
    cf.dec_dropout = dec_dropout
    cf.reload_decoder_state = reload_decoder_state

    cf.layer_wise_attn = layer_wise_attn
    cf.attn_type = attn_type
    cf.force_copy = force_copy

    cf.lr = lr
    cf.sort_by_len = sort_by_len
    cf.test_batch_size = test_batch_size
    cf.warmup_epochs = warmup_epochs
    cf.epochs = epochs
    cf.batch_size = batch_size
    cf.aim = aim
    cf.print_copy_info = print_copy_info

    output_path = os.path.join("../output", cf.dataset_type)
    make_directory(output_path)
    cf.model_save_path = os.path.join(output_path, "model.pth")
    cf.pred_file = os.path.join(output_path, "predict.json")
    cf.log_file = os.path.join(output_path, 'logging.txt' if not cf.only_test else 'logging_test.txt')

    cf.early_stop = early_stop
    cf.checkpoint = checkpoint
    cf.clip = clip

    cf.summary_vocab_size = summary_vocab_size
    cf.code_vocab_size = code_vocab_size
    cf.ast_vocab_size = ast_vocab_size
    cf.enc_layers = enc_layers
    cf.dec_layers = dec_layers
    cf.RvNN_input_dim = RvNN_input_dim
    cf.aggregate_dim = aggregate_dim

    cf.d_k = cf.code_embedding_dim // cf.nhead
    cf.d_v = cf.code_embedding_dim // cf.nhead
    cf.d_ff = d_ff
    cf.use_asts = use_asts
    cf.initialize_weights = initialize_weights
    cf.copy_attn = copy_attn
    cf.lr_decay = lr_decay

    javadoc_extension = "original"
    code_extension = "subtoken"
    dataset_path = cf.dataset_path
    data_dir = os.path.join(dataset_path, "code_sum")
    cf.train_src_files = os.path.join(data_dir, "train/code.%s" % code_extension)
    cf.train_tgt_files = os.path.join(data_dir, "train/javadoc.%s" % javadoc_extension)

    cf.dev_src_files = os.path.join(data_dir, "val/code.%s" % code_extension)
    cf.dev_tgt_files = os.path.join(data_dir, "val/javadoc.%s" % javadoc_extension)

    cf.test_src_files = os.path.join(data_dir, "test/code.%s" % code_extension)
    cf.test_tgt_files = os.path.join(data_dir, "test/javadoc.%s" % javadoc_extension)

    cf.ast_original_token_wc = os.path.join(dataset_path, "ASTs/asts_word_count.pkl")
    cf.train_ast_files = os.path.join(dataset_path, "ASTs/train/split_AST.pkl")
    cf.dev_ast_files = os.path.join(dataset_path, "ASTs/val/split_AST.pkl")
    cf.test_ast_files = os.path.join(dataset_path, "ASTs/test/split_AST.pkl")

    cf.train_rebuild_tree_files = os.path.join(dataset_path, "ASTs/train/rebuild_tree.pkl")
    cf.dev_rebuild_tree_files = os.path.join(dataset_path, "ASTs/val/rebuild_tree.pkl")
    cf.test_rebuild_tree_files = os.path.join(dataset_path, "ASTs/test/rebuild_tree.pkl")
    cf.big_graph_path = os.path.join(dataset_path, "ASTs/fids.pkl")

    if cf.only_test:
        cf.dev_src_files = cf.test_src_files
        cf.dev_tgt_files = cf.test_tgt_files
        cf.dev_ast_files = cf.test_ast_files
        cf.dev_rebuild_tree_files = cf.test_rebuild_tree_files
        cf.is_DDP = False

    cf.activate_f = activate_f


def get_model_bindings(idx):
    with open(Config.gin_config_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data[idx]


class Config:
    gin_config_json_file = "diff_n_head_of_c2nl_transformer.json"
    gin_config_path = 'hyperparameter_setting/%s' % gin_config_json_file


def set_seed(SEED=0):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx', type=int, default=0, required=False)
    parser.add_argument("-only_test", default=False, required=False)
    parser.add_argument("-cf_file", type=str, default="TL_CodeSum.json", required=False)
    parser.add_argument("-ddp", required=False)
    parser.add_argument("--local_rank", type=int, default=-1, required=False)
    parser.add_argument('-bs', type=int, required=False)

    args_cmd = parser.parse_args()
    cf.only_test = args_cmd.only_test
    set_seed(0)
    Config.gin_config_path = 'hyperparameter_setting/%s' % args_cmd.cf_file
    model_bindings = get_model_bindings(args_cmd.idx)
    gin.parse_config_files_and_bindings([], model_bindings)
    set_config()
    if args_cmd.ddp:
        cf.is_DDP = True if "True" in args_cmd.ddp else False
    if args_cmd.bs:
        cf.batch_size = int(args_cmd.bs)

    logger = set_logger(DEBUG=False, log_file=cf.log_file + str(args_cmd.local_rank) + ".txt", checkpoint=cf.checkpoint)
    if cf.is_DDP:
        torch.distributed.init_process_group(backend="nccl")
    main(cf)
