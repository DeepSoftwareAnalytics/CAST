#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import os
import copy
import time
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from multiprocessing import cpu_count, Pool
import sys

sys.path.append("../util")
from Config import Config as cf
from LoggerUtil import set_logger, debug_logger
from DataUtil import read_pickle_data, tree_to_index, save_pickle_data, make_directory, time_format
import re
from spiral import ronin


def find_root(graph, child):
    parent = list(graph.predecessors(child))
    if len(parent) == 0:
        return child
    else:
        return find_root(graph, parent[0])


# Given a graph including some subtree
# return a list of root nodes  and  a list of subtrees
def get_subtrees(graph):
    root = []
    subtrees = {}
    all_nodes = copy.deepcopy(set(graph.nodes()))
    idx = 1
    number_nodes = len(all_nodes)
    while idx <= number_nodes:
        node_idx = "n" + str(idx)
        node_idx = find_root(graph, node_idx)
        root.append(node_idx)
        tree = nx.dfs_tree(graph, node_idx)
        subtrees[node_idx] = nx.dfs_successors(graph, node_idx)
        subtree_nodes = set(tree.nodes())
        idx += len(subtree_nodes)
    return root, subtrees


def processing_node(s):
    s = s.strip()
    pat = r"'[\s\S]*'"
    s = re.sub(pat, "<STR>", s)
    res = [tok.lower() for tok in ronin.split(s) if tok]
    # return res
    return [[item] for item in res]


def format_node(node_label):
    node_label_tokens = node_label.split(": ")

    if node_label_tokens[0] == "INIT" and len(node_label_tokens) > 1:
        label_tokens = node_label_tokens[1].split(" ____ ")
        init_label_token = label_tokens[0].split()
        init_label_token.extend(label_tokens[1:])
        node_data = [[token.strip()] for token in init_label_token if token]
        node_data.insert(0, "INIT")
    elif node_label_tokens[0] in Config.type_set and len(node_label_tokens) > 1:
        label_tokens = node_label_tokens[1].split(" ____ ")
        node_data = [[token.strip()] for token in label_tokens if token]
        node_data.insert(0, node_label_tokens[0])
    else:
        label_tokens = node_label.split(" ____ ")
        if len(label_tokens) > 1:
            node_data = [[token] for token in label_tokens if token]
            node_data.insert(0, "STMT")
        else:
            node_data = [node_label]
    return node_data


def format_node_subtoken(node_label):
    node_label_tokens = node_label.split(": ")

    if node_label_tokens[0] == "INIT" and len(node_label_tokens) > 1:
        label_tokens = node_label_tokens[1].split(" ____ ")
        init_label_token = label_tokens[0].split()
        init_label_token.extend(label_tokens[1:])
        # node_data = [[token.strip()] for token in init_label_token if token]
        # node_data = [processing_node(token) for token in init_label_token if token][0]
        # node_data.insert(0, "INIT")
        node_data = ["INIT"]
        for token in init_label_token:
            if token:
                node_data.extend(processing_node(token))

    elif node_label_tokens[0] in Config.type_set and len(node_label_tokens) > 1:
        label_tokens = node_label_tokens[1].split(" ____ ")
        # node_data = [[token.strip()] for token in label_tokens if token]
        # node_data = [processing_node(token) for token in label_tokens if token][0]
        # node_data.insert(0, node_label_tokens[0])
        node_data = [node_label_tokens[0]]
        for token in label_tokens:
            if token:
                node_data.extend(processing_node(token))
    else:
        label_tokens = node_label.split(" ____ ")
        if len(label_tokens) > 1:
            # node_data = [[token] for token in label_tokens if token]
            # node_data = [processing_node(token) for token in label_tokens if token][0]
            # node_data.insert(0, "STMT")
            node_data = ["STMT"]
            for token in label_tokens:
                if token:
                    node_data.extend(processing_node(token))
        else:
            node_data = [node_label]
            # node_data = processing_node(node_label)
    return node_data


# trees : a -> b -> c; b->d; x->y->w,y->z;
# root: a, x
# subtrees_with_label [[a,[b,[c],[d]]],  [x,[y ,[w], [z]]]]
def convert_dict_to_list(root_node, tree, tree_with_label, node_info):
    children = tree[root_node]
    for r in children:
        if tree_with_label[0] == "ROOT":
            node = format_node(node_info[r]["label"][1:-1])
        else:
            node = format_node_subtoken(node_info[r]["label"][1:-1])
        # if len(node) > 1:
        #     tree_with_label.extend(node)
        # else:
        tree_with_label.append(node)
        if r in tree.keys():
            convert_dict_to_list(r, tree, tree_with_label[-1], node_info)
        else:
            continue


# converting some trees  to  the list
# for example
# trees : a -> b -> c; b->d; x->y->w,y->z;
# root: a, x
# subtrees_with_label [[a,[b,[c],[d]]],  [x,[y ,[w], [z]]]]
def convert_tree(root, trees, node_data):
    subtrees_with_node_label = []
    for r in root:
        subtree = trees[r]
        node_label = node_data[r]["label"][1:-1]
        node_label_token = node_label.split(":")
        if len(node_label_token) > 1:
            subtrees_with_node_label.append(["ROOT"])
        else:
            subtrees_with_node_label.append([node_label])

        convert_dict_to_list(r, subtree, subtrees_with_node_label[-1], node_data)

    return subtrees_with_node_label


def get_splitted_ast(fid):
    dot_file = os.path.join(cf.dot_files_dir, str(fid) + "-AST.dot")
    ast_graph = nx.DiGraph(read_dot(dot_file))
    node_information = dict(ast_graph.nodes(True))
    root_nodes, subtrees = get_subtrees(ast_graph)
    if not root_nodes:
        debug_logger("root_nodes is none %d " % fid)
        return None
    try:
        converted_subtrees = convert_tree(root_nodes, subtrees, node_information)
        return converted_subtrees
    # except KeyError and IndexError:
    except KeyError and IndexError:
        debug_logger("converted_subtrees %d " % fid)
        return None


class Config:
    is_multi_processing = True
    type_set = ["MODIFIER", "RETURN", "NAME", "TYPE", "UPDATE", "IN", "CASE", "COND"]
    # use_codebert_data = True
    use_codebert_data = False


if __name__ == '__main__':
    log_file = "./log/4_1_get_splitted_ast_subtoken.txt"
    set_logger(cf.debug, log_file, checkpoint=True)

    # correct_splitted_ast_fid = read_pickle_data(os.path.join(cf.correct_fid, "correct_splitted_ast_fid.pkl"))
    # correct_splitted_ast_fid = read_pickle_data(os.path.join(cf.correct_fid, "correct_splitted_ast_fid.pkl"))
    # correct_splitted_ast_fid = read_pickle_data(os.path.join(cf.correct_fid, "codebert_fids.pkl"))

    # correct_fid = read_pickle_data(os.path.join(cf.correct_fid, "finally_summary_id.pkl"))

    correct_fid = read_pickle_data(os.path.join(cf.correct_fid, "correct_splitted_ast_fid.pkl"))
    if Config.use_codebert_data:
        codebert_fid = read_pickle_data(os.path.join(cf.correct_fid, "codebert_fids.pkl"))
        for part in correct_fid:
            correct_fid[part] = list(list(set(correct_fid[part]).intersection(set(codebert_fid[part]))))

    correct_splitted_ast_fid = correct_fid

    parts = ['train', 'test', 'val']

    asts_path = os.path.join(cf.data_root_path, 'ASTs')
    make_directory(asts_path)

    debug_logger("start to get splitted ast list")

    correct_parsed_splitted_ast_fid = {}
    start = time.perf_counter()
    if Config.is_multi_processing:
        cores = cpu_count()
        for part in ["val", "test", "train"]:
            # for part in ["test"]:
            pool = Pool(cores)
            results = pool.map(get_splitted_ast, correct_splitted_ast_fid[part])
            pool.close()
            pool.join()
            correct_parsed_splitted_ast_fid[part] = [fid for i, fid in enumerate(correct_splitted_ast_fid[part]) if
                                                     results[i]]
            asts_tree = {fid: results[i] for i, fid in enumerate(correct_splitted_ast_fid[part]) if results[i]}
            debug_logger("%s Done" % part)
            debug_logger(" time cost :" + time_format(time.perf_counter() - start))
            save_pickle_data(os.path.join(asts_path, part), "sliced_AST_subtoken.pkl", asts_tree)
            del asts_tree

        asts_tree_train = read_pickle_data(os.path.join(asts_path, "train", "sliced_AST_subtoken.pkl"))
        asts_tree_val = read_pickle_data(os.path.join(asts_path, "val", "sliced_AST_subtoken.pkl"))
        asts_tree_test = read_pickle_data(os.path.join(asts_path, "test", "sliced_AST_subtoken.pkl"))
        asts_tree = {"train": asts_tree_train, "val": asts_tree_val, "test": asts_tree_test}
    else:
        asts_tree = {}

        for part in parts:
            asts_tree[part] = {}
            correct_parsed_splitted_ast_fid[part] = []
            for fid in correct_splitted_ast_fid[part]:
                asts_part = get_splitted_ast(fid)
                asts_tree[part][fid] = asts_part
                if asts_part:
                    correct_parsed_splitted_ast_fid[part].append(fid)

    debug_logger(" time cost :" + time_format(time.perf_counter() - start))
    save_pickle_data(asts_path, 'sliced_AST_subtoken.pkl', asts_tree)
    # save_pickle_data(cf.correct_fid, 'correct_parsed_splitted_ast_fid.pkl', correct_parsed_splitted_ast_fid)

    # asts_tree1 = read_pickle_data(os.path.join(asts_path, 'sliced_AST_list2.pkl'))
    # print("asts_tree  ", asts_tree1 == asts_tree)
