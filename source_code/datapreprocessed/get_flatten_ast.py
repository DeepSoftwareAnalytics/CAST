#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import os
import time
import sys
from multiprocessing import cpu_count, Pool

sys.path.append("../util")
from Config import Config as cf
from LoggerUtil import set_logger, debug_logger
from DataUtil import read_pickle_data, tree_to_index, save_pickle_data, make_directory, time_format


def preOrderTraverse(tree, sequence):
    if not tree:
        return None
    #     print(tree[0])
    sequence.append(tree[0])
    for subtree in tree[1:]:
        preOrderTraverse(subtree, sequence)


def get_splitted_ast_sequence(tree):
    sequence = []
    for subtree in tree:
        preOrderTraverse(subtree, sequence)
    return sequence


class Config:
    is_multi_processing = True
    use_subtoken_dataset = True


if __name__ == '__main__':
    # set_logger(cf.debug, None, None)
    log_file = "./log/5_get_flatten_ast.txt"
    set_logger(cf.debug, log_file, checkpoint=True)
    debug_logger("start to get flatten ast")
    asts_path = os.path.join(cf.data_root_path, 'ASTs')
    if Config.use_subtoken_dataset:
        asts_tree = read_pickle_data(os.path.join(asts_path, 'sliced_AST_subtoken.pkl'))
    else:
        asts_tree = read_pickle_data(os.path.join(asts_path, 'sliced_AST.pkl'))
    start = time.perf_counter()
    if Config.is_multi_processing:
        cores = cpu_count()
        pool = Pool(cores)
        results = pool.map(get_splitted_ast_sequence, list(asts_tree["train"].values()))
        pool.close()
        pool.join()
        asts_sequence_corpus = {fid: results[i] for i, fid in enumerate(asts_tree["train"].keys()) if results[i]}
    else:
        asts_sequence_corpus = {}
        for fid, trees in asts_tree["train"].items():
            flatten_tree = get_splitted_ast_sequence(trees)
            asts_sequence_corpus[fid] = flatten_tree
    if Config.use_subtoken_dataset:
        save_pickle_data(asts_path, 'asts_sequence_subtoken_corpus.pkl', asts_sequence_corpus)
    else:
        save_pickle_data(asts_path, 'asts_sequence_corpus.pkl', asts_sequence_corpus)
    debug_logger(" time cost :" + time_format(time.perf_counter() - start))

    # asts_sequence_corpus1 = read_pickle_data(os.path.join(asts_path, 'asts_sequence_corpus1.pkl'))
    # print("asts_tree  ", asts_sequence_corpus1 == asts_sequence_corpus)
