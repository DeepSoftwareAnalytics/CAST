#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import math
import nltk
from javalang.ast import Node
import javalang
import sys
import subprocess
import numpy as np
# sys.path.append(r"F:\AST_slice_project\AST_slice\source_code\util\Tree.py")
sys.setrecursionlimit(10000)
# from util.Tree import  BlockNode
sys.path.append("../")
from util.Config import Config as cf
import pickle
import os
# from gensim.models.word2vec import Word2Vec
import copy
from torch.utils.data import DataLoader
from util.Dataset import CodeSummaryDataset, collate_function, SplittedASTDataset, collate_function_splitted_AST
from torch.utils.data.distributed import DistributedSampler
from util.LoggerUtil import info_logger
from multiprocessing import cpu_count, Pool
import re
import json
from gensim.models.word2vec import Word2Vec


# from spiral import ronin


# copy from https://github.com/zhangj111/astnn/blob/master/clone/utils.py
def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'  # node.pop()
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token


# copy from https://github.com/zhangj111/astnn/blob/master/clone/utils.py
def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


# copy from https://github.com/zhangj111/astnn/blob/master/clone/utils.py
def get_sequence(node, sequence):
    token, children = get_token(node), get_children(node)
    sequence.append(token)

    for child in children:
        get_sequence(child, sequence)

    if token in ['ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement']:
        sequence.append('End')


# https://github.com/zhangj111/astnn/blob/master/clone/utils.py#L49-L78
def get_blocks(node, block_seq):
    name, children = get_token(node), get_children(node)
    logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
    if name in ['MethodDeclaration', 'ConstructorDeclaration']:
        block_seq.append(BlockNode(node))
        body = node.body
        for child in body:
            if get_token(child) not in logic and not hasattr(child, 'block'):
                block_seq.append(BlockNode(child))
            else:
                get_blocks(child, block_seq)
    elif name in logic:
        block_seq.append(BlockNode(node))
        for child in children[1:]:
            token = get_token(child)
            if not hasattr(node, 'block') and token not in logic + ['BlockStatement']:
                block_seq.append(BlockNode(child))
            else:
                get_blocks(child, block_seq)
            block_seq.append(BlockNode('End'))
    elif name is 'BlockStatement' or hasattr(node, 'block'):
        block_seq.append(BlockNode(name))
        for child in children:
            if get_token(child) not in logic:
                block_seq.append(BlockNode(child))
            else:
                get_blocks(child, block_seq)
    else:
        for child in children:
            get_blocks(child, block_seq)


# https://github.com/zhangj111/astnn/blob/master/clone/pipeline.py#L102-L105
def trans_to_sequences(ast):
    sequence = []
    get_sequence(ast, sequence)
    return sequence


# https://github.com/zhangj111/astnn/blob/master/clone/pipeline.py
# ast_token 2 index
def tree_to_index(node, vocab, max_token):
    token = node.token
    # result = [vocab[token].index if token in vocab else max_token]
    # result = [token if token in vocab else max_token]
    result = [token]
    children = node.children
    for child in children:
        result.append(tree_to_index(child, vocab, max_token))
    return result


# https://github.com/zhangj111/astnn/blob/master/clone/pipeline.py
def trans2seq(r, vocab, max_token):
    blocks = []
    get_blocks(r, blocks)
    tree = []
    for b in blocks:
        btree = tree_to_index(b, vocab, max_token)
        tree.append(btree)
    return tree


def get_position_of_subtree(node, positions):
    token = node.token
    # result = [vocab[token].index if token in vocab else max_token]
    # result = [token if token in vocab else max_token]
    result = [token]
    children = node.children
    try:
        positions.append(node.position)
    except:
        pass
    for child in children:
        result.append(get_position_of_subtree(child, positions))
    return result


def stmt_slicing(r, vocab, max_token):
    blocks = []
    get_blocks(r, blocks)
    tree = {}
    previous_position = 0
    for b in blocks:
        positions = []
        btree = get_position_of_subtree(b, positions)
        # tree.append(btree)
        positions = set(positions)
        positions = positions - set([-100])
        positions = list(positions)

        if len(positions) == 0:
            tree[previous_position + 1] = btree
            previous_position += 1
            continue

        previous_position = max(positions)
        tree[positions[0]] = btree
        if len(positions) == 1:
            continue
        for position in positions[1:]:
            tree[position] = positions[0]
    return tree


# https://github.com/zhangj111/astnn/blob/master/clone/pipeline.py#L35-L39
def parse_program(method_code):
    tokens = javalang.tokenizer.tokenize(method_code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


# load the data from the data_path
def read_pickle_data(data_path):
    print("read pickle from %s" % (data_path))
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


#  Make a new directory if it is not exist.
def make_directory(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    else:
        pass


# Write data to save_path/file_name.pkl file
def write_to_pickle(save_path, data, file_name='new_sample_files.pkl'):
    make_directory(save_path)
    path = os.path.join(save_path, file_name)
    print("save dataset in " + path)
    with open(path, 'wb') as output:
        pickle.dump(data, output)


# given [a, [ b ,[c] ,[d]]] return [90,[ 23 ,[45] ,[13]]]
def tree_to_index(trees, word2vec):
    vocab = word2vec.vocab
    # try:
    for i in range(len(trees)):
        try:
            node = trees[i]
        except:
            pass
        if type(node) == str:
            # trees[i] = word2idx(node,vocab)
            trees[i] = vocab[node].index if node in vocab else word2vec.syn0.shape[0]
        else:
            tree_to_index(node, word2vec)
    # except:
    #     print("")
    # returntrees


def array_split(original_data, core_num):
    data = []
    total_size = len(original_data)
    per_core_size = math.ceil(total_size / core_num)
    for i in range(core_num):
        lower_bound = i * per_core_size
        upper_bound = min((i + 1) * per_core_size, total_size)
        data.append(original_data[lower_bound:upper_bound])
    return data


def time_form(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def stem_and_replace_predict_target(predict, target):
    predict_copy = copy.deepcopy(predict)
    target_copy = copy.deepcopy(target)
    empty_cnt = 0
    new_predict = []
    new_target = []
    stemming = nltk.stem.SnowballStemmer('english')

    for i in range(len(predict_copy)):
        pred = copy.deepcopy(predict_copy[i])
        if len(pred) < 1:
            empty_cnt += 1
            continue
        tar = copy.deepcopy(target_copy[i][0])
        try:
            pred[0] = stemming.stem(pred[0])
            tar[0] = stemming.stem(tar[0])
        except:
            continue
        if pred[0] == "get":
            pred[0] == "return"
        if tar[0] == "get":
            tar[0] == "return"
        new_predict.append(pred)
        new_target.append([tar])
    print("empty_cnt", empty_cnt)
    return new_predict, new_target


def get_file_name():
    data_len_str = "dlen" + str(cf.code_max_len) + "_clen" + str(cf.sum_max_len) + "_slen" + str(cf.asts_len)
    vocab_size_str = "_dvoc" + str(cf.code_vocab_size) + "_cvoc" + str(cf.summary_vocab_size) \
                     + "_svoc" + str(cf.ast_vocab_size)
    # word_cnt_str = "_dwc" + str(cf.code_min_count) + "_cwc" + str(cf.summary_min_count) \
    #                   + "_swc" + str(cf.ast_token_min_count)
    approach_str = "_approach_" + cf.approach + "_is_rebuild_" + str(cf.is_rebuild_tree + 0)

    if cf.pretrained_model_type != "None":
        pretrain_str = "pre_train_mt_" + cf.cf.pretrained_model_type
        filename = data_len_str + vocab_size_str + approach_str + pretrain_str + '_dataset.pkl'
        pass
    elif cf.approach == "astnn":
        filename = data_len_str + vocab_size_str + approach_str + "_astvoc" + str(cf.ast_vocab_size) + '_dataset.pkl'
    else:
        filename = data_len_str + vocab_size_str + approach_str + '_dataset.pkl'
    return filename


def get_config_str():
    config_str = "activate_f" + str(cf.activate_f) + "_lr" + str(cf.lr) + "_batch_size" + str(cf.batch_size) + \
                 "_Clip" + str(cf.clip) + "_enc_layers" + str(cf.enc_layers) + "_dec_layers" + str(cf.dec_layers) \
                 + "_nhead" + str(cf.nhead) + "_d_ff" + str(cf.d_ff) + "_node_embedding_dim" + \
                 str(cf.node_embedding_dim) + "_summary_embedding_dim" + str(cf.summary_embedding_dim)

    return config_str


#  word cont based
# def read_funcom_format_data(path):
#     data = read_pickle_data(path)
#
#     # load summary
#     train_summary = data["ctrain"]
#     val_summary = data["cval"]
#     test_summary = data["ctest"]
#
#     # sliced ast
#     train_asts = data["strain"]
#     val_asts = data["sval"]
#     test_asts = data["stest"]
#
#     # load code
#     train_code = None
#     val_code = None
#     test_code = None
#     if cf.model_type == 'RvNNCodeAttGRU':
#         train_code = data["dtrain"]
#         val_code = data["dval"]
#         test_code = data["dtest"]
#
#     # fids
#     val_ids = list(data["cval"].keys())
#     test_ids = list(data['ctest'].keys())
#
#     # vocabulary info
#     # summary_vocab = data["comstok"]
#     # code_vocab = data["datstok"]
#
#     # summary vocab info
#     # i2w info
#     summary_token_i2w = data["comstok"]["i2w"]
#     # code_token_i2w = code_vocab["i2w"]
#
#     summary_vocab_size = data["config"]["comvocabsize"]
#     cf.UNK_token_id = summary_vocab_size - 4
#     code_vocab_size = data["config"]["datvocabsize"]
#
#     cf.summary_vocab_size = summary_vocab_size
#     cf.PAD_token_id = summary_vocab_size - 3     # <NULL>
#     cf.SOS_token_id = summary_vocab_size - 2     # <s>
#     cf.EOS_token_id = summary_vocab_size - 1    # </s>
#
#     # code summary info
#     code_vocab_size = data["config"]["datvocabsize"]
#     cf.code_vocab_size = code_vocab_size
#
#     # Asts vocab info
#     asts_vocab_size = data["config"]["astsvocabsize"]
#     cf.ast_vocab_size = asts_vocab_size
#
#     summary_len = data["config"]["comlen"]
#     # obtain DataLoader for iteration
#     train_dataset = CodeSummaryDataset(summary=train_summary, asts=train_asts, code=train_code)
#     val_dataset = CodeSummaryDataset(summary=val_summary, asts=val_asts, code=val_code)
#     test_dataset = CodeSummaryDataset(summary=test_summary, asts=test_asts, code=test_code)
#     if cf.is_DDP:
#         train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size, collate_fn=collate_function,
#                                        pin_memory=True, num_workers=cf.num_subprocesses,
#                                        sampler=DistributedSampler(train_dataset), drop_last=True)
#     else:
#         train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size, collate_fn=collate_function,
#                                        pin_memory=True,num_workers=cf.num_subprocesses, drop_last=True)
#     # val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
#     #                    num_workers=cf.num_subprocesses, sampler=DistributedSampler(val_dataset), drop_last=True)
#     # test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
#     #                        num_workers=cf.num_subprocesses, sampler=DistributedSampler(test_dataset), drop_last=True)
#     val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
#                                  pin_memory=True, num_workers=cf.num_subprocesses,  drop_last=True)
#     test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
#                                   pin_memory=True, num_workers=cf.num_subprocesses, drop_last=True)
#     return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, \
#         code_vocab_size, asts_vocab_size, summary_vocab_size, summary_token_i2w, summary_len, val_ids, test_ids


# vocabsize based
def read_funcom_format_data(path):
    data = read_pickle_data(path)

    # load summary
    train_summary = data["ctrain"]
    val_summary = data["cval"]
    test_summary = data["ctest"]

    # sliced ast
    train_asts = data["strain"]
    val_asts = data["sval"]
    test_asts = data["stest"]

    # load code
    train_code = None
    val_code = None
    test_code = None
    if cf.model_type == 'RvNNCodeAttGRU' or cf.model_type == 'RvNNCodeAttPooling' or cf.model_type == 'RvNNCodeAttTrans' or cf.model_type == "RvNNCodeAstSumTrans":
        train_code = data["dtrain"]
        val_code = data["dval"]
        test_code = data["dtest"]

    # fids
    val_ids = list(data["cval"].keys())
    test_ids = list(data['ctest'].keys())

    # summary vocab info
    # i2w info
    summary_token_i2w = data["comstok"]["i2w"]
    # code_token_i2w = code_vocab["i2w"]

    summary_vocab_size = data["config"]["comvocabsize"]
    cf.summary_vocab_size = summary_vocab_size
    code_vocab_size = data["config"]["datvocabsize"]
    cf.code_vocab_size = code_vocab_size
    # Asts vocab info
    asts_vocab_size = data["config"]["astsvocabsize"]

    cf.ast_vocab_size = asts_vocab_size
    # if cf.approach == "astnn":
    #     cf.ast_vocab_size = asts_vocab_size + 1
    summary_len = data["config"]["comlen"]
    cf.UNK_token_id = summary_vocab_size - 1
    # obtain DataLoader for iteration
    train_dataset = CodeSummaryDataset(summary=train_summary, asts=train_asts, code=train_code)
    val_dataset = CodeSummaryDataset(summary=val_summary, asts=val_asts, code=val_code)
    test_dataset = CodeSummaryDataset(summary=test_summary, asts=test_asts, code=test_code)
    if cf.is_DDP:
        # train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size, collate_fn=collate_function,
        #                                pin_memory=True, num_workers=cf.num_subprocesses,
        #                                sampler=DistributedSampler(train_dataset), drop_last=True)
        train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size, collate_fn=collate_function,
                                       pin_memory=False, num_workers=cf.num_subprocesses,
                                       sampler=DistributedSampler(train_dataset), drop_last=True)
    else:
        train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size, collate_fn=collate_function,
                                       pin_memory=False, num_workers=cf.num_subprocesses, drop_last=True)
    # val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
    #                    num_workers=cf.num_subprocesses, sampler=DistributedSampler(val_dataset), drop_last=True)
    # test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
    #                        num_workers=cf.num_subprocesses, sampler=DistributedSampler(test_dataset), drop_last=True)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
                                 pin_memory=False, num_workers=cf.num_subprocesses, drop_last=True)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
                                  pin_memory=False, num_workers=cf.num_subprocesses, drop_last=True)

    # pretrained_embedding = None
    # if cf.pretrained_model_type != "None":
    #     pretrained_embedding =data["aststok"]["pre_train_embed"]
    #     cf.pretrained_weight = pretrained_embedding

    return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, \
           code_vocab_size, asts_vocab_size, summary_vocab_size, summary_token_i2w, summary_len, val_ids, test_ids


def read_format_data_with_rebuild_tree(path):
    data = read_pickle_data(path)

    # load summary
    train_summary = data["ctrain"]
    val_summary = data["cval"]
    test_summary = data["ctest"]

    # sliced ast
    train_asts = data["strain"]
    val_asts = data["sval"]
    test_asts = data["stest"]

    # rebuild_tree
    train_rebuild_tree = data["rtrain"]
    val_rebuild_tree = data["rval"]
    test_rebuild_tree = data["rtest"]

    # load code
    # train_code = None
    # val_code = None
    # test_code = None
    # if cf.model_type == 'RvNNCodeAttGRU' or cf.model_type == 'RvNNCodeAttGRU' :
    train_code = data["dtrain"]
    val_code = data["dval"]
    test_code = data["dtest"]

    # fids
    val_ids = list(data["cval"].keys())
    test_ids = list(data['ctest'].keys())

    # summary vocab info
    # i2w info
    summary_token_i2w = data["comstok"]["i2w"]
    # code_token_i2w = code_vocab["i2w"]

    summary_vocab_size = data["config"]["comvocabsize"]
    cf.summary_vocab_size = summary_vocab_size
    code_vocab_size = data["config"]["datvocabsize"]
    cf.code_vocab_size = code_vocab_size
    # Asts vocab info
    asts_vocab_size = data["config"]["astsvocabsize"]

    cf.ast_vocab_size = asts_vocab_size
    # if cf.approach == "astnn":
    #     cf.ast_vocab_size = asts_vocab_size + 1
    summary_len = data["config"]["comlen"]
    cf.UNK_token_id = summary_vocab_size - 1
    # obtain DataLoader for iteration
    train_dataset = SplittedASTDataset(summary=train_summary, asts=train_asts, rebuild_tree=train_rebuild_tree,
                                       code=train_code)
    val_dataset = SplittedASTDataset(summary=val_summary, asts=val_asts, rebuild_tree=val_rebuild_tree, code=val_code)
    test_dataset = SplittedASTDataset(summary=test_summary, asts=test_asts, rebuild_tree=test_rebuild_tree,
                                      code=test_code)
    if cf.is_DDP:
        # train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size, collate_fn=collate_function,
        #                                pin_memory=True, num_workers=cf.num_subprocesses,
        #                                sampler=DistributedSampler(train_dataset), drop_last=True)
        train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size,
                                       collate_fn=collate_function_splitted_AST,
                                       pin_memory=False, num_workers=cf.num_subprocesses,
                                       sampler=DistributedSampler(train_dataset), drop_last=True)
    else:
        train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size,
                                       collate_fn=collate_function_splitted_AST,
                                       pin_memory=False, num_workers=cf.num_subprocesses, drop_last=True)
    # val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
    #                    num_workers=cf.num_subprocesses, sampler=DistributedSampler(val_dataset), drop_last=True)
    # test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, collate_fn=collate_function,
    #                        num_workers=cf.num_subprocesses, sampler=DistributedSampler(test_dataset), drop_last=True)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size,
                                 collate_fn=collate_function_splitted_AST,
                                 pin_memory=False, num_workers=cf.num_subprocesses, drop_last=True)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size,
                                  collate_fn=collate_function_splitted_AST,
                                  pin_memory=False, num_workers=cf.num_subprocesses, drop_last=True)

    # pretrained_embedding = None
    # if cf.pretrained_model_type != "None":
    #     pretrained_embedding =data["aststok"]["pre_train_embed"]
    #     cf.pretrained_weight = pretrained_embedding

    return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, \
           code_vocab_size, asts_vocab_size, summary_vocab_size, summary_token_i2w, summary_len, val_ids, test_ids


def delete_error_fids(all_fids, deleted_fids):
    if type(deleted_fids) != dict:
        deleted_fids = {'train': deleted_fids, 'test': deleted_fids, 'val': deleted_fids}
    correct_fids = {}
    parts = ['train', 'test', 'val']
    for part in parts:
        # print("get_correct_fid.")
        correct_fids[part] = []
        fids = all_fids[part]
        for fid in fids:
            if not (fid in deleted_fids[part]):
                correct_fids[part].append(fid)
    return correct_fids


def basic_info_logger():
    # info_logger("[Setting] EXP: %s" % (str(cf.EXP)))
    # info_logger("[Setting] DEBUG: %s" % (str(cf.DEBUG)))
    info_logger("[Setting] trim_til_EOS: %s" % (str(cf.trim_til_EOS)))
    info_logger("[Setting] use_full_sum: %s" % (str(cf.use_full_sum)))
    # info_logger("[Setting] use_oov_sum: %s" % (str(cf.use_oov_sum)))
    info_logger("[Setting] Method: %s" % cf.model_type)
    info_logger("[Setting] dataset_path: %s" % cf.dataset_path)
    info_logger("[Setting] GPU id: %d" % cf.gpu_id)
    info_logger("[Setting] num_epochs: %d" % cf.epochs)
    info_logger("[Setting] batch_size: %d" % cf.batch_size)
    info_logger("[Setting] ast_vocab_size: %d" % cf.ast_vocab_size)
    info_logger("[Setting] summary_vocab_size: %d" % cf.summary_vocab_size)
    info_logger("[Setting] asts_embedding_dim): %d" % cf.node_embedding_dim)
    info_logger("[Setting] summary_embedding_dim: %d" % cf.summary_embedding_dim)
    info_logger("[Setting] biGRU_hidden_size: %d" % cf.bigru_hidden_dim)
    info_logger("[Setting] decoder_rnn_hidden_size: %d" % cf.decoder_rnn_hidden_size)
    info_logger("[Setting] sum_max_len : %d" % cf.sum_max_len)
    info_logger("[Setting] pretrained_weight : %s" % cf.pretrained_weight)
    info_logger("[Setting] lr: %f" % cf.lr)
    info_logger("[Setting] weight_decay : %f" % cf.weight_decay)
    # info_logger("[Setting] num_subprocesses: %d" % cf.num_subprocesses)
    # info_logger("[Setting] eval_frequency: %d" % cf.eval_frequency)
    # info_logger("[Setting] out_path: %s" % cf.out_path)


def lower_sequence(sequences):
    lower_seq = {}
    for idx, items in sequences.items():
        sub_sequence = [tok.lower() for tok in items]
        lower_seq[idx] = sub_sequence
    return lower_seq


def print_time(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    print("time_cost: %02d:%02d:%02d" % (h, m, s))


# copy from CoCoGUM
def code_tokens_filter_punctuation(sequence):
    tokens = []
    for s in sequence:
        # https://www.jianshu.com/p/4f476942dca8
        s = re.sub('\W+', '', s).replace("_", '')
        if s:
            tokens.append(s)
    return tokens


# copy from CoCoGUM
def code_tokens_filter_punctuation_lists(seqs):
    return [code_tokens_filter_punctuation(seq) for seq in seqs]


# copy from CoCoGUM
def code_tokens_filter_punctuation_parallel(data):
    cores = cpu_count() - 1
    pool = Pool(cores)
    ids = list(data.keys())
    ids_split = array_split(ids, cores)
    data_split = []

    for split in ids_split:
        data_split.append([data[i] for i in split])

    results = pool.map(code_tokens_filter_punctuation_lists, data_split)
    new_data = {}
    for ids, result in zip(ids_split, results):
        for i in range(len(ids)):
            mid = ids[i]
            new_data[mid] = result[i]
    pool.close()
    pool.join()
    return new_data


# copy from CoCoGUM
def code_tokens_replace_str_num(sequence):
    tokens = []
    for s in sequence:
        if s[0] == '"' and s[-1] == '"':
            tokens.append("<STRING>")
        elif s.isdigit():
            tokens.append("<NUM>")
        else:
            tokens.append(s)
    return tokens


# copy from CoCoGUM
def code_tokens_replace_str_num_lists(seqs):
    return [code_tokens_replace_str_num(seq) for seq in seqs]


# copy from CoCoGUM
def code_tokens_replace_str_num_parallel(data):
    cores = cpu_count() - 1
    pool = Pool(cores)
    ids = list(data.keys())
    ids_split = array_split(ids, cores)
    data_split = []

    for split in ids_split:
        data_split.append([data[i] for i in split])

    results = pool.map(code_tokens_replace_str_num_lists, data_split)
    new_data = {}
    for ids, result in zip(ids_split, results):
        for i in range(len(ids)):
            mid = ids[i]
            new_data[mid] = result[i]
    pool.close()
    pool.join()
    return new_data


def delete_error_fids(all_fids, deleted_fids):
    if type(deleted_fids) != dict:
        deleted_fids = {'train': deleted_fids, 'test': deleted_fids, 'val': deleted_fids}
    correct_fids = {}
    parts = ['train', 'test', 'val']
    for part in parts:
        # print("get_correct_fid.")
        correct_fids[part] = []
        fids = all_fids[part]
        for fid in fids:
            if not (fid in deleted_fids[part]):
                correct_fids[part].append(fid)
    return correct_fids


def str_to_bool(str_data):
    return True if str_data.lower() == 'true' else False


# copy from CoCoGUM
def save_pickle_data(path_dir, filename, data):
    full_path = path_dir + '/' + filename
    print("Save dataset to: %s" % full_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    with open(full_path, 'wb') as output:
        pickle.dump(data, output)


# copy from CoCoGUM
def get_all_tokens(data):
    # data is a dict: {idx: string_seq, ....}
    all_tokens = []
    for seq in data.values():
        # tokens = seq.split(" ")
        # tokens = list(filter(lambda x: x,   tokens))
        all_tokens.extend(seq)
    return all_tokens


def get_asts_tokens(data):
    # data is a dict: {idx: string_seq, ....}
    # ['ROOT:', '0.java', 'METHOD_SIGNATURWE']
    all_tokens = []
    for seq in data:
        del seq[1]  # seq[1] is **.java
        all_tokens.extend(seq)
    return all_tokens


# copy from CoCoGUM
# if vocab_size is negative, all tokens will be used
def build_vocab(word_count, start_id, vocab_size=-1):
    w2i, i2w = {}, {}
    # word_count_ord[i][0] -> word, word_count_ord[i][1] -> count
    word_count_ord = sorted(word_count.items(), key=lambda item: item[1], reverse=True)

    if vocab_size > 0:
        print("vocab_size (exclude special tokens)", vocab_size)
        size = vocab_size
    else:
        size = len(word_count_ord)
        print("use all tokens ", size)

    for i in range(size):
        w2i[word_count_ord[i][0]] = i + start_id
        i2w[i + start_id] = word_count_ord[i][0]

    return w2i, i2w


# copy from CoCoGUM
def build_vocab_with_pad_unk(word_count, start_id, vocab_size=-1):
    w2i, i2w = build_vocab(word_count, start_id, vocab_size)

    w2i[cf.PAD_token] = cf.PAD_token_id
    i2w[cf.PAD_token_id] = cf.PAD_token

    unk_id = len(w2i)
    w2i[cf.UNK_token] = unk_id
    i2w[unk_id] = cf.UNK_token
    return w2i, i2w


def build_vocab_with_pad_unk_sos_eos(word_count, start_id, vocab_size=-1):
    w2i, i2w = build_vocab(word_count, start_id, vocab_size)
    w2i[cf.SOS_token] = cf.SOS_token_id
    i2w[cf.SOS_token_id] = cf.SOS_token
    w2i[cf.EOS_token] = cf.EOS_token_id
    i2w[cf.EOS_token_id] = cf.EOS_token
    w2i[cf.PAD_token] = cf.PAD_token_id
    i2w[cf.PAD_token_id] = cf.PAD_token

    unk_id = len(w2i)
    w2i[cf.UNK_token] = unk_id
    i2w[unk_id] = cf.UNK_token
    return w2i, i2w


# copy from CoCoGUM
def build_vocab_info(code_word_frequency, summary_word_frequency, ast_word_frequency):
    # build code vocab, include <NULL> and <UNK>. Start after <NULL>=0.
    code_w2i, code_i2w = build_vocab_with_pad_unk(code_word_frequency, 1, cf.code_vocab_size - 2)
    # build summary vocab, include <s>, </s>, <NULL> and <UNK>. Start after </s>=2.
    summary_w2i, summary_i2w = build_vocab_with_pad_unk_sos_eos(summary_word_frequency, 3, cf.summary_vocab_size - 4)
    # build code vocab, include <NULL> and <UNK>. Start after <NULL>=0.
    ast_w2i, ast_i2w = build_vocab_with_pad_unk(ast_word_frequency, 1, cf.ast_vocab_size - 2)

    vocab = {'code_word_count': code_word_frequency,
             'summary_word_count': summary_word_frequency,
             # 'sbt_word_count': sbt_word_frequency,
             'ast_word_count': ast_word_frequency,
             'summary_w2i': summary_w2i,
             'summary_i2w': summary_i2w,
             # 'ast_i2w': ast_i2w,
             'ast_i2w': ast_i2w,
             'code_i2w': code_i2w,
             'code_w2i': code_w2i,
             # 'ast_w2i': ast_w2i}
             'ast_w2i': ast_w2i}
    return vocab


# copy from CoCoGUM
def save_json_data(path_dir, filename, data):
    make_directory(path_dir)
    path = os.path.join(path_dir, filename)
    print("save  dataset  in " + path)
    with open(path, 'w') as output:
        json.dump(data, output)


# copy from CoCoGUM
def filter_vocab_size(full_w2i, vocab_size):
    w2i = {}
    i2w = {}
    if vocab_size > 0:
        sort_w2i = sorted(full_w2i.items(), key=lambda item: item[1])[:vocab_size - 1]
    else:
        sort_w2i = sorted(full_w2i.items(), key=lambda item: item[1])[:len(full_w2i) - 1]

    for word, idx in sort_w2i:
        w2i[word] = idx
        i2w[idx] = word
    unk_id = len(w2i)
    w2i[cf.UNK_token] = unk_id
    i2w[unk_id] = cf.UNK_token
    return w2i, i2w


# copy from CoCoGUM
def load_vocab_(vocab_path, vocab_size={}):
    with open(vocab_path, 'r') as f:
        vocab_info = json.load(f)
    code_word_count = vocab_info['code_word_count']
    summary_word_count = vocab_info['summary_word_count']
    sbt_word_count = vocab_info['ast_word_count']

    summary_w2i, summary_i2w = \
        filter_vocab_size(vocab_info['summary_w2i'], vocab_size.get("summary", -1))
    code_w2i, code_i2w = \
        filter_vocab_size(vocab_info['code_w2i'], vocab_size.get("code", -1))
    sbt_w2i, sbt_i2w = \
        filter_vocab_size(vocab_info['ast_w2i'], vocab_size.get("ast", -1))

    return code_word_count, summary_word_count, sbt_word_count, \
           summary_w2i, summary_i2w, code_i2w, code_w2i, \
           sbt_i2w, sbt_w2i


# copy from CoCoGUM
def load_sum_and_code_vocab(vocab_path, vocab_size={}):
    with open(vocab_path, 'r') as f:
        vocab_info = json.load(f)
    code_word_count = vocab_info['code_word_count']
    summary_word_count = vocab_info['summary_word_count']
    # sbt_word_count = vocab_info['ast_word_count']

    summary_w2i, summary_i2w = \
        filter_vocab_size(vocab_info['summary_w2i'], vocab_size.get("summary", -1))
    code_w2i, code_i2w = \
        filter_vocab_size(vocab_info['code_w2i'], vocab_size.get("code", -1))
    # sbt_w2i,  sbt_i2w = \
    #     filter_vocab_size(vocab_info['ast_w2i'], vocab_size.get("ast", -1))

    return code_word_count, summary_word_count, \
           summary_w2i, summary_i2w, code_i2w, code_w2i, \
 \
        # copy from CoCoGUM


def padding(line, max_len, padding_id):
    line_len = len(line)
    if line_len < max_len:
        line += [padding_id] * (max_len - line_len)
    return line


# copy from CoCoGUM
def code2ids(tokens, w2i, seq_len):
    unk_id = w2i[cf.UNK_token]
    ids = [w2i.get(token, unk_id) for token in tokens[:seq_len]]
    ids = padding(ids, seq_len, cf.PAD_token_id)
    return ids


# copy from CoCoGUM
def summary2ids(summary_tokens, summary_w2i):
    sum_max_len = cf.sum_max_len
    summary_unk_id = summary_w2i[cf.UNK_token]
    summary_ids = [summary_w2i.get(token, summary_unk_id) for token in summary_tokens[:sum_max_len - 1]]
    summary_ids.insert(0, cf.SOS_token_id)
    if len(summary_ids) < sum_max_len:
        summary_ids.append(cf.EOS_token_id)
    summary_ids = padding(summary_ids, sum_max_len, cf.PAD_token_id)
    return summary_ids


# given [a, [ b ,[c] ,[d]]] return [90,[ 23 ,[45] ,[13]]]
def tree2idx(trees, w2i):
    # vocab = word2vec.vocab
    # try:
    max_len = len(w2i) - 1
    for i in range(len(trees)):
        try:
            node = trees[i]
        except:
            pass
        if type(node) == str:
            # trees[i] = word2idx(node,vocab)
            # trees[i] = vocab[node].index if node in vocab else word2vec.syn0.shape[0]
            trees[i] = w2i[node] if node in w2i else max_len
        else:
            tree2idx(node, w2i)


# copy from CoCoGUM
def process_data(summary_tokens, code_tokens_javalang, ast_tokens, code_w2i, summary_w2i, ast_w2i, correct_fid):
    method_code = {}
    method_summary = {}
    method_ast = {}

    for fid in correct_fid:
        try:
            method_code_fid = code2ids(code_tokens_javalang[fid], code_w2i, cf.code_max_len)
            method_summary_fid = summary2ids(summary_tokens[fid], summary_w2i)
            # tree2idx(ast_tokens_index[fid], ast_w2i)
            tree2idx(ast_tokens[fid], ast_w2i)
            method_ast_fid = ast_tokens[fid]

            # method_ast[fid] = tree2ids(sbt_tokens[fid], sbt_w2i, cf.sbt_len)
        except:
            print("wrong fid:", fid)
            continue
        method_code[fid] = method_code_fid
        method_summary[fid] = method_summary_fid
        method_ast[fid] = method_ast_fid
        # method_code[fid] = code2ids(code_tokens_javalang[fid], code_w2i, cf.code_max_len )
        # method_summary[fid] = summary2ids(summary_tokens[fid], summary_w2i)
        # tree2idx(ast_tokens[fid], ast_w2i)
        # method_ast[fid] = ast_tokens[fid]
        # method_ast[fid] = tree2ids(sbt_tokens[fid], sbt_w2i, cf.sbt_len)
    return method_code, method_summary, method_ast


def process_data_with_rebulid_tree(summary_tokens, code_tokens_javalang, ast_tokens, code_w2i, summary_w2i, ast_w2i,
                                   correct_fid, rebuild_tree):
    method_code = {}
    method_summary = {}
    method_ast = {}
    method_rebuild_tree = {}

    for fid in correct_fid:
        try:
            method_code_fid = code2ids(code_tokens_javalang[fid], code_w2i, cf.code_max_len)
            method_summary_fid = summary2ids(summary_tokens[fid], summary_w2i)
            # tree2idx(ast_tokens_index[fid], ast_w2i)
            tree2idx(ast_tokens[fid], ast_w2i)
            method_ast_fid = ast_tokens[fid]
            method_rebuild_tree_fid = rebuild_tree[fid]
            # method_ast[fid] = tree2ids(sbt_tokens[fid], sbt_w2i, cf.sbt_len)
        except:
            print("wrong fid:", fid)
            continue
        method_code[fid] = method_code_fid
        method_summary[fid] = method_summary_fid
        method_ast[fid] = method_ast_fid
        method_rebuild_tree[fid] = method_rebuild_tree_fid
        # method_code[fid] = code2ids(code_tokens_javalang[fid], code_w2i, cf.code_max_len )
        # method_summary[fid] = summary2ids(summary_tokens[fid], summary_w2i)
        # tree2idx(ast_tokens[fid], ast_w2i)
        # method_ast[fid] = ast_tokens[fid]
        # method_ast[fid] = tree2ids(sbt_tokens[fid], sbt_w2i, cf.sbt_len)
    return method_code, method_summary, method_ast, method_rebuild_tree


# copy from CoCoGUM
def process_data_(summary_tokens, code_tokens_javalang, ast_tokens_index, code_w2i, summary_w2i, correct_fid):
    method_code = {}
    method_summary = {}
    method_ast = {}

    for fid in correct_fid:
        try:
            method_code_fid = code2ids(code_tokens_javalang[fid], code_w2i, cf.code_max_len)
            method_summary_fid = summary2ids(summary_tokens[fid], summary_w2i)
            # tree2idx(ast_tokens_index[fid], ast_w2i)
            method_ast_fid = ast_tokens_index[fid]
            # method_ast[fid] = tree2ids(sbt_tokens[fid], sbt_w2i, cf.sbt_len)
        except:
            print("wrong fid:", fid)
            continue
        method_code[fid] = method_code_fid
        method_summary[fid] = method_summary_fid
        method_ast[fid] = method_ast_fid
    return method_code, method_summary, method_ast


def load_asts_vocab_astnn(dataset_path):
    astnn_data_path = os.path.join(dataset_path, 'astnn_data')
    # ast_index_file_name = 'java_method_stmt_trees_index.pkl'
    w2v_path = os.path.join(astnn_data_path,
                            'node_w2v_dim_' + str(cf.astnn_w2v_size) + "vocabsize_" + str(cf.ast_vocab_size))
    ast_node_word2vec = Word2Vec.load(w2v_path).wv
    ast_i2w = ast_node_word2vec.index2word
    ast_vocab = ast_node_word2vec.vocab  # index and word_count

    return ast_i2w, ast_vocab


def update_dict(d, f):
    """
    Update dict d = {k:v} with f, return {k:f(v)}
    :param d:
    :param f:
    :return:
    """
    cores = cpu_count()
    pool = Pool(cores)
    keys = []
    values = []
    for i in list(d):
        keys.append(i)
        values.append(d[i])
    results = pool.map(f, values)
    pool.close()
    pool.join()
    new_d = {}
    for k, v in zip(keys, results):
        new_d[k] = v
    return new_d


def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    return "%02d:%02d:%02d" % (h, m, s)


def get_object_attr(obj):
    return '\n'.join(['%s:%s' % item for item in obj.__dict__.items()])


def process_sbt_token(token, xml_tokens, index, node_type):
    continue_flag = False
    terminal_node_flag = False

    # Replacing '<type' with '(type' for non-terminal
    # Replacing '<\type' with 'type)' for non-terminal
    if node_type == "non_terminal":
        if token[:2] == '</':
            new_token = token.replace('</', ') ')
        else:
            new_token = token.replace('<', '( ')
    elif node_type == "terminal":
        new_token = ') ' + xml_tokens[index - 1].replace('<', '') + '_' + token
        terminal_node_flag = True
    else:
        literal_type = {
            'number': '<NUM>',
            'string': '<STR>',
            'null': 'null',
            'char': '<STR>',
            'boolean': token
        }
        new_token = ') ' + xml_tokens[index - 2].replace('<', '') + '_' + node_type + '_' + literal_type[
            node_type]
        terminal_node_flag = True

    return new_token, terminal_node_flag


def verify_node_type(token, xml_tokens, index, literal_list):
    """
    non_terminal:
        <type>
        </type>
    terminal:
        <type> token </type>  (this is right)
        <type> token <value> ... ( this is wrong)
    literal:
        <literal type="String" token </literal> | number | char | string | null | boolean
    """
    try:
        if token[0] == '<':
            node_type = 'non_terminal'
        elif xml_tokens[index - 1][1:] == xml_tokens[index + 1][2:] and xml_tokens[index - 1][0] == '<':
            node_type = 'terminal'
        elif xml_tokens[index - 1][:5] == 'type=':
            token_type = xml_tokens[index - 1].replace('type=', '')
            token_type = token_type.replace('\"', '')
            if token_type in literal_list:
                node_type = token_type
            else:
                node_type = None
        else:
            node_type = None
        return node_type
    except IndexError as e:
        print(e)
        return None


# Given the AST generating by http://131.123.42.38/lmcrs/beta/ ,
# it return SBT proposed by https://xin-xia.github.io/publication/icpc182.pdf
def xml2sbt(xml):
    # Replacing '<...>' with ' <...'
    xml = xml.replace('<', ' <')
    xml = xml.replace('>', ' ')

    #  splitting xml and filtering ''
    xml_tokens = xml.split(' ')
    xml_tokens = [i for i in xml_tokens if i != '']

    sbt = []
    terminal_node_flag = False
    literal_list = ['number', 'string', 'null', 'char', 'boolean']
    for i in range(len(xml_tokens)):

        # i = i+1 is unavailable in for loop, so we set terminal_node_flag to skip
        # terminal_nodes that have already been processed
        if terminal_node_flag:
            terminal_node_flag = False
            continue
        token = xml_tokens[i]
        node_type = verify_node_type(token, xml_tokens, i, literal_list)
        if node_type:
            new_token, terminal_node_flag = process_sbt_token(token, xml_tokens, i, node_type)
            sbt.append(new_token)
        else:
            continue
    return sbt


# Obtaining the sbt of the java file
def sbt_parser(file_path):
    if os.path.isfile(file_path):
        commandline = 'srcml ' + file_path
    else:
        commandline = 'srcml -l Java -t "{}"'.format(file_path)
    # https://docs.python.org/3/library/subprocess.html
    # Window
    # xml, _ = subprocess.Popen(commandline, stdout=subprocess.PIPE,
    # stderr=subprocess.PIPE).communicate(timeout=20)
    # ubutu
    xml, _ = subprocess.Popen(commandline.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(
        timeout=10)
    try:
        xml = re.findall(r"(<unit .*</unit>)", xml.decode('utf-8'), re.S)[0]
    except:
        print(xml)
        xml = " "
    sbt = xml2sbt(xml)
    return ' '.join(sbt)


# Write to java file
def write_source_code_to_java_file(path, method_id, method):
    java_path = os.path.join(path, str(method_id) + ".java")
    with open(java_path, "w") as f:
        f.write(method)
    return java_path

def percent_len(all_len, title="code"):
    percentiles = np.array([20, 25, 30, 40, 50,  60, 70, 75, 80, 90, 95, 100])
    # Compute percentiles: ptiles_vers
    ptiles_vers = np.percentile(all_len, percentiles)
    # Print the result
    print("percent %s legth" % (title))
    for percentile, ptiles_ver in zip(percentiles, ptiles_vers):
        print(percentile, "\t", ptiles_ver)
    print("mean", "\t", np.mean(all_len))