#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import os
import time
import sys
from multiprocessing import cpu_count, Pool

sys.path.append("../util")
from Config import Config as cf
from LoggerUtil import set_logger, debug_logger
from DataUtil import read_pickle_data, tree_to_index, save_pickle_data, make_directory, time_format, percent_len
import numpy as np


def preOrderTraverse(tree, sequence):
    if not tree:
        return None
    #     print(tree[0])
    sequence.append(tree[0])
    for subtree in tree[1:]:
        preOrderTraverse(subtree, sequence)


def get_splitted_ast_sequence(tree):
    sequences = []
    for subtree in tree:
        sequence = []
        preOrderTraverse(subtree, sequence)
        sequences.append(sequence)
    return sequences


def get_flatted_sequence():
    debug_logger("start to get flatten ast")
    asts_path = os.path.join(cf.data_root_path, 'ASTs')
    if Config.use_subtoken_dataset:
        asts_tree = read_pickle_data(os.path.join(asts_path, 'sliced_AST_subtoken.pkl'))
    else:
        asts_tree = read_pickle_data(os.path.join(asts_path, 'sliced_AST.pkl'))
    start = time.perf_counter()
    flatten_ast = {}
    if Config.is_multi_processing:
        for part in asts_tree:
            cores = cpu_count()
            pool = Pool(cores)
            results = pool.map(get_splitted_ast_sequence, list(asts_tree[part].values()))
            pool.close()
            pool.join()
            flatten_ast[part] = {fid: results[i] for i, fid in enumerate(asts_tree[part].keys()) if results[i]}
    else:
        for part in asts_tree:
            flatten_ast[part] = {}
            for fid, trees in asts_tree[part].items():
                flatten_tree = get_splitted_ast_sequence(trees)
                flatten_ast[part][fid] = flatten_tree

    save_pickle_data(asts_path, 'flatten_ast.pkl', flatten_ast)
    debug_logger(" time cost :" + time_format(time.perf_counter() - start))


def count_tree_length(tree):
    tree_length = {}
    for part in tree:
        tree_length[part] = {}
        for fid in tree[part]:
            tree_length[part][fid] = [len(seq) for seq in tree[part][fid]]
    return tree_length


def get_big_graph(tree_length):
    tree_len_cnt_gt_1000 = {}
    for part in tree_length:
        tree_len_cnt_gt_1000[part] = [fid for fid in tree_length[part] if
                                      max(tree_length[part][fid]) > Config.max_node_num]
    return tree_len_cnt_gt_1000


class Config:
    is_multi_processing = True
    max_node_num = 500
    use_subtoken_dataset = True


# def percent_len(all_len, title="code"):
#     percentiles = np.array([20, 25, 30, 40, 50,  60, 70, 75, 80, 90, 95, 100])
#     # Compute percentiles: ptiles_vers
#     ptiles_vers = np.percentile(all_len, percentiles)
#     # Print the result
#     print("percent %s legth" % (title))
#     for percentile, ptiles_ver in zip(percentiles, ptiles_vers):
#         print(percentile, "\t", ptiles_ver)
#     print("mean", "\t", np.mean(all_len))

if __name__ == '__main__':
    # set_logger(cf.debug, None, None)
    log_file = "./log/5_1_get_big_graph_fid.txt"
    set_logger(cf.debug, log_file, checkpoint=True)
    is_flatten_ast_exit = False
    start = time.perf_counter()
    flatten_ast_path = os.path.join(cf.data_root_path, 'ASTs', 'flatten_ast.pkl')
    if os.path.exists(flatten_ast_path):
        is_flatten_ast_exit = True
    if not is_flatten_ast_exit:
        get_flatted_sequence()

    flatten_ast = read_pickle_data(flatten_ast_path)
    flatten_ast_length = count_tree_length(flatten_ast)
    full_ast_len = [np.sum(item) for item in flatten_ast_length["train"].values()]
    # print( "----------------%s--------------"%cf.dataset_type)
    # print("The number of AST  node  before split")
    # percent_len(full_ast_len, title="ast")
    split_ast_len = []
    split_ast_cnt = []
    for item in flatten_ast_length["train"].values():
        split_ast_len.extend(item)
        split_ast_cnt.append(len(item))
    # print("The number of AST  node   after split split")
    # percent_len(split_ast_len, title="ast")
    # print("The number of subtrees   after split split")
    # percent_len(split_ast_cnt, title="ast")

    tree_len_cnt_gt_1000_fid = get_big_graph(flatten_ast_length)
    for part in tree_len_cnt_gt_1000_fid:
        print(part, len(tree_len_cnt_gt_1000_fid[part]))
    save_pickle_data(cf.correct_fid, 'tree_len_cnt_gt_%s_fid.pkl' % str(Config.max_node_num), tree_len_cnt_gt_1000_fid)
    debug_logger(" time cost :" + time_format(time.perf_counter() - start))
    # # check
    # dataset1 = read_pickle_data(os.path.join(cf.correct_fid, "tree_len_cnt_gt_1000_fid1.pkl"))
    # print("dataset1 == dataset ", dataset1 == tree_len_cnt_gt_1000_fid)
