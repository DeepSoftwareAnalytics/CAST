#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import sys
import os
import time
from collections import Counter
from multiprocessing import cpu_count, Pool

sys.path.append("../util")
from Config import Config as cf
from DataUtil import read_pickle_data, save_pickle_data, array_split, save_json_data


def get_all_tokens(data):
    # data is a dict: {idx: string_seq, ....}
    all_tokens = []
    for seq in data.values():
        all_tokens.extend(seq)
    return all_tokens


def count_word_parallel(word_list):
    cores = cpu_count()
    pool = Pool(cores)
    word_split = array_split(word_list, cores)
    word_counts = pool.map(Counter, word_split)
    result = Counter()
    for wc in word_counts:
        result += wc
    pool.close()
    pool.join()
    return dict(result.most_common())  # return the dict sorted by frequency reversely.


def get_word_frequence(token_path):
    # calculate token_word_count on train set
    start = time.perf_counter()
    # tokens = read_pickle_data(token_path)
    csn = read_pickle_data(os.path.join(cf.data_root_path, "csn.pkl"))
    tokens= {part: {fid: csn[part][fid]["code_tokens"] for fid in csn[part]} for part in csn}
    word_list = get_all_tokens(tokens["train"])
    print("get all tokens: time cost", time.perf_counter() - start)

    start = time.perf_counter()
    token_word_count = count_word_parallel(word_list)
    print("count_word_parallel time cost", time.perf_counter() - start)
    print("token_word_count length: ", len(token_word_count))
    return token_word_count


def word_count_asts(token_path):
    # calculate token_word_count on train set
    start = time.perf_counter()
    tokens = read_pickle_data(token_path)
    word_list = get_all_tokens(tokens)
    print("get all tokens: ", time.perf_counter() - start)

    start = time.perf_counter()
    token_word_count = count_word_parallel(word_list)
    print("count_asts_word_parallel time cost", time.perf_counter() - start)
    print("token_word_count length: ", len(token_word_count))
    return token_word_count


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
             'ast_word_count': ast_word_frequency,
             'summary_w2i': summary_w2i,
             'summary_i2w': summary_i2w,
             'ast_i2w': ast_i2w,
             'code_i2w': code_i2w,
             'code_w2i': code_w2i,
             'ast_w2i': ast_w2i}
    return vocab


def summary_process(data_path, voc_path):
    cwc = True
    # cwc:summary word count
    summary_tokens_file_name = Config.summary_tokens_file_name
    if os.path.exists(os.path.join(voc_path, "summary_word_count_" + summary_tokens_file_name)):
        cwc = False
    if cwc:
        print("build vocabulary of summary")
        summary_tokens_path = os.path.join(data_path, "summary", summary_tokens_file_name)
        summary_word_count = get_word_frequence(summary_tokens_path)
        save_pickle_data(voc_path, "summary_word_count_" + summary_tokens_file_name, summary_word_count)


def code_process(data_path, voc_path):
    dwc = True
    code_tokens_file_name = Config.code_tokens_file_name
    if os.path.exists(os.path.join(voc_path, "code_word_count_" + code_tokens_file_name)):
        dwc = False

    if dwc:
        print("build vocabulary of  code")
        code_tokens_path = os.path.join(data_path, "code", code_tokens_file_name)
        code_word_count = get_word_frequence(code_tokens_path)
        save_pickle_data(voc_path, "code_word_count_" + code_tokens_file_name, code_word_count)


def ast_process(data_path, voc_path):
    swc = True
    if Config.use_subtoken_dataset:
        if os.path.exists(os.path.join(voc_path, 'asts_subtoken_word_count.pkl')):
            swc = False
    else:
        if os.path.exists(os.path.join(voc_path, 'asts_word_count.pkl')):
            swc = False
    if swc:
        print("build vocabulary of  asts")
        if Config.use_subtoken_dataset:
            ast_tokens_path = os.path.join(data_path, "ASTs", 'asts_sequence_subtoken_corpus.pkl')
            asts_word_count = word_count_asts(ast_tokens_path)
            save_pickle_data(voc_path, "asts_subtoken_word_count.pkl", asts_word_count)
        else:
            ast_tokens_path = os.path.join(data_path, "ASTs", 'asts_sequence_corpus.pkl')
            asts_word_count = word_count_asts(ast_tokens_path)
            save_pickle_data(voc_path, "asts_word_count.pkl", asts_word_count)


class Config:
    summary_tokens_file_name = "cfp1_csi1_cfd0_clc1.pkl"
    code_tokens_file_name = "djl1_dfp0_dsi1_dlc1_dr1.pkl"
    use_subtoken_dataset =True


def word_count_and_get_vocabulary():
    data_path = cf.data_root_path
    path = os.path.join(data_path, "vocab_raw")
    # -----build vocabulary of asts---------
    ast_process(data_path, path)
#     code_process(data_path, path)

#     return


#     if Config.use_subtoken_dataset:

#         asts_word_count = read_pickle_data(os.path.join(path, 'asts_subtoken_word_count.pkl'))
#     else:
#         # -----build vocabulary of summary---------
#         summary_process(data_path, path)
#         # -----build vocabulary of code---------
#         code_process(data_path, path)
#         summary_tokens_file_name = Config.summary_tokens_file_name
#         code_tokens_file_name = Config.code_tokens_file_name
#         summary_word_count = read_pickle_data(os.path.join(path, "summary_word_count_" + summary_tokens_file_name))
#         code_word_count = read_pickle_data(os.path.join(path, "code_word_count_" + code_tokens_file_name))
#         asts_word_count = read_pickle_data(os.path.join(path, 'asts_word_count.pkl'))

#         start = time.perf_counter()
#         vocab_info = build_vocab_info(code_word_count, summary_word_count, asts_word_count)
#         voc_info_file_name = 'csn_trainingset_' + \
#                              code_tokens_file_name.split(".")[0] + "_" + \
#                              summary_tokens_file_name.split(".")[0] + '.json'
#         save_json_data(path, voc_info_file_name, vocab_info)
#         print("build csn vocabulary time cost: ", time.perf_counter() - start)


if __name__ == '__main__':
    cf.ast_vocab_size = 10000  # RvNNCodeAttn astnn
    cf.summary_vocab_size = 10000
    cf.code_vocab_size = 10000
    word_count_and_get_vocabulary()
