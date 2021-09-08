#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import pickle
import time
import os
from multiprocessing import cpu_count, Pool

import sys

sys.path.append("../util")
from Config import Config as cf
from DataUtil import save_pickle_data, read_pickle_data, time_format
from LoggerUtil import set_logger, debug_logger


def tree_to_seq_with_parent_position(trees, parent_position, seq, position):
    if trees[0] == 'METHOD_BODY':
        pass
    else:
        seq.append(trees[0])
        position.append(parent_position)
        parent_position = len(position)
    for i in range(1, len(trees)):
        node = trees[i]
        tree_to_seq_with_parent_position(node, parent_position, seq, position)


def postorder_traversal(trees, seq):
    for i in range(1, len(trees)):
        node = trees[i]
        if len(node) == 1:
            seq.append(node[0])
        else:
            postorder_traversal(node, seq)
    if type(trees[0]) == str:
        seq.append(trees[0])


def process_main_body(main_body):
    position = []
    seq = []
    parent_position = 0
    tree_to_seq_with_parent_position(main_body, parent_position, seq, position)
    return seq, position


def process_splitted_tree(subtrees):
    # subtrees = sliced_AST
    subtrees_list = []
    subtrees_root_node_list = []
    # trees_root_node_list =  subtrees_root_node_list

    for subtree in subtrees:
        subtrees_root_node_list.append(subtree[0])
        seq = []
        postorder_traversal(subtree, seq)
        subtrees_list.append(seq)
    return subtrees_list, subtrees_root_node_list


def get_parent_of_node(main_body_seq, main_body_parent, root_list, subtrees_list):
    tree_parent = main_body_parent[:2]
    main_body_point = 2
    subtrees_root_node_list_point = 2
    try:
        for root_node in root_list[2:]:

            if main_body_point >= len(main_body_seq):
                break
            if root_node == main_body_seq[main_body_point]:
                tree_parent.append(main_body_parent[main_body_point])
                main_body_point += 1
                subtrees_root_node_list_point += 1
            if main_body_point < len(main_body_seq):
                if main_body_seq[main_body_point] == "STATIC-BLOCK":
                    main_body_point += 1
                else:
                    main_boy_node_index = root_list.index(main_body_seq[main_body_point],
                                                          subtrees_root_node_list_point)
            if root_node[:7] == "NESTED_":
                subtrees_root_node_list_point += 1
                if root_node in subtrees_list[subtrees_root_node_list_point][:-1]:
                    tree_parent.append(subtrees_root_node_list_point + 1)
                else:
                    tree_parent.append(main_boy_node_index + 1)
        return tree_parent
    except Exception as e:
        # print("wrong fid: ", fid)
        # print("main_body_seq: ", main_body_seq)
        # print("main_body_parent: ", main_body_parent)
        # print("subtrees_root_node_list: ", root_list)
        # Config.wrong_cnt += 1
        # print("wrong_cnt: ", Config.wrong_cnt)
        print(str(e))
        return None


def get_tree_format(tree_parent):
    tree_parent_index = [p - 1 for p in tree_parent]
    rebuild_tree = {}
    for i, parent in enumerate(tree_parent_index[1:]):
        if parent in rebuild_tree.keys():
            rebuild_tree[parent].append(i + 1)
        else:
            rebuild_tree[parent] = [i + 1]
    return rebuild_tree


def rebuild_structure_tree(sliced_AST):
    # main body
    main_body_seq, main_body_parent = process_main_body(sliced_AST[0])

    # sliced AST
    subtrees_list, subtrees_root_node_list = process_splitted_tree(sliced_AST)

    # root nodes set -> sliced_tree_parent.
    sliced_tree_parent = get_parent_of_node(main_body_seq, main_body_parent, subtrees_root_node_list, subtrees_list)

    # rebuild tree
    if sliced_tree_parent:
        rebuild_tree = get_tree_format(sliced_tree_parent)
        return rebuild_tree
    else:
        return None


class Config:
    is_multi_processing = True
    wrong_cnt = 0


if __name__ == '__main__':
    log_file = "./log/6_get_rebuild_tree.txt"
    set_logger(cf.debug, log_file, checkpoint=True)
    debug_logger("start to get rebuild tree")
    start = time.perf_counter()
    asts_path = os.path.join(cf.data_root_path, 'ASTs')
    # path = os.path.join(asts_path, "sliced_AST.pkl")
    path = os.path.join(asts_path, "sliced_AST_subtoken.pkl")
    sliced_AST_list = pickle.load(open(path, "rb"))
    correct_rebuild_tree_fid = {}
    if Config.is_multi_processing:
        cores = cpu_count()
        for part in ["val", "train", "test"]:
            pool = Pool(cores)
            results = pool.map(rebuild_structure_tree, list(sliced_AST_list[part].values()))
            pool.close()
            pool.join()
            correct_rebuild_tree_fid[part] = [fid for i, fid in enumerate(sliced_AST_list[part].keys()) if
                                              results[i]]
            rebuild_tree_part = {fid: results[i] for i, fid in enumerate(sliced_AST_list[part].keys()) if results[i]}
            debug_logger("%s Done" % part)
            debug_logger(" time cost :" + time_format(time.perf_counter() - start))
            save_pickle_data(os.path.join(asts_path, part), "rebuild_tree.pkl", rebuild_tree_part)
            del rebuild_tree_part

        asts_tree_train = read_pickle_data(os.path.join(asts_path, "train", "rebuild_tree.pkl"))
        asts_tree_val = read_pickle_data(os.path.join(asts_path, "val", "rebuild_tree.pkl"))
        asts_tree_test = read_pickle_data(os.path.join(asts_path, "test", "rebuild_tree.pkl"))
        rebuild_tree_list = {"train": asts_tree_train, "val": asts_tree_val, "test": asts_tree_test}

    else:
        rebuild_tree_list = {}
        for part in ['train', 'test', 'val']:
            rebuild_tree_list_part = {}
            correct_rebuild_tree_fid[part] = []
            for fid in sliced_AST_list[part]:
                rebuild_tree_list_part[fid] = rebuild_structure_tree(sliced_AST_list[part][fid])
                if rebuild_tree_list_part[fid]:
                    correct_rebuild_tree_fid[part].append(fid)

            rebuild_tree_list[part] = rebuild_tree_list_part

    save_pickle_data(asts_path, 'rebuild_tree.pkl', rebuild_tree_list)
    save_pickle_data(cf.correct_fid, 'correct_rebuild_tree_fid.pkl', correct_rebuild_tree_fid)
    for part in correct_rebuild_tree_fid:
        debug_logger(" %s length: %d " % (part, len(correct_rebuild_tree_fid[part])))
    debug_logger(" time cost :" + time_format(time.perf_counter() - start))

    # rebuild_tree_list1 = read_pickle_data(os.path.join(asts_path, 'rebuild_tree_list2.pkl'))
    # print("asts_tree  ", rebuild_tree_list1 == rebuild_tree_list)
