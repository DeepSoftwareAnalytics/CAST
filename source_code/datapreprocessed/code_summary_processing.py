# !/usr/bin/env python
# !-*-coding:utf-8 -*- 

import sys
import os
import time
import pickle
import javalang

sys.path.append("../util")
from Config import Config as cf
from DataUtil import write_to_pickle, read_pickle_data, tree_to_index, lower_sequence, print_time, \
    code_tokens_filter_punctuation_parallel, code_tokens_replace_str_num_parallel, make_directory
from DataUtil_the_thrid_pkg import code_tokens_split_identifier_parallel


def gen_tokens_lowercase(data):
    print(30 * "-")
    print("gen_tokens lower_identifier")
    code_tokens_lower = {}
    start_time = time.perf_counter()
    for partition in cf.parts:
        partition_lower = lower_sequence(data[partition])
        code_tokens_lower[partition] = partition_lower
    print_time(time.perf_counter() - start_time)
    return code_tokens_lower


def gen_code_tokens_filter_punctuation(dataset_javalang):
    print(30 * "-")
    print("gen_tokens_filter_punctuation")
    code_tokens_filter = {}
    start_time = time.perf_counter()
    for partition in cf.parts:
        partition_split = code_tokens_filter_punctuation_parallel(dataset_javalang[partition])
        code_tokens_filter[partition] = partition_split
    print_time(time.perf_counter() - start_time)
    return code_tokens_filter


def gen_tokens_split_identifier(data):
    print(30 * "-")
    print("gen_tokens_split_identifier")
    code_tokens_split = {}
    start_time = time.perf_counter()
    for partition in cf.parts:
        partition_split = code_tokens_split_identifier_parallel(data[partition])
        code_tokens_split[partition] = partition_split
    print_time(time.perf_counter() - start_time)
    return code_tokens_split


def gen_code_tokens_replace_str_num(dataset_javalang):
    print(30 * "-")
    print("gen_tokens_replace_str_num")
    code_tokens_replace = {}
    start_time = time.perf_counter()
    for partition in cf.parts:
        partition_replace = code_tokens_replace_str_num_parallel(dataset_javalang[partition])
        code_tokens_replace[partition] = partition_replace
    print_time(time.perf_counter() - start_time)
    return code_tokens_replace


if __name__ == '__main__':

    data_path = cf.data_root_path
    cf.parts = ["val", "train", "test"]
    parts = cf.parts
    correct_rebuild_tree_fid = read_pickle_data(os.path.join(cf.correct_fid, 'correct_rebuild_tree_fid.pkl'))
    csn = pickle.load(open(os.path.join(data_path, '../../Data/TL_CodeSum/csn.pkl'), "rb"))

    # ----------------summary processing -------------------------------
    summary_path = os.path.join(data_path, 'summary')
    make_directory(summary_path)

    # ---------summary extraction------------
    cf.extract_summary = True
    if cf.extract_summary:
        print("start extract_summary")
        start_time = time.perf_counter()
        summaries = {}
        for part in parts:
            summaries[part] = {}
            for fid in correct_rebuild_tree_fid[part]:
                docstring_tokens = csn[part][fid]['docstring_tokens']
                summaries[part][fid] = docstring_tokens
        write_to_pickle(summary_path, summaries, file_name='cfp0_csi0_cfd0_clc0.pkl')
        print("extract summary time cost: ", time.perf_counter() - start_time)

    # ---------summary_tokens lower case-------------
    cf.is_clc = True
    if cf.is_clc:
        path = cf.data_root_path
        docstring_token = pickle.load(open(os.path.join(path, 'summary/cfp0_csi0_cfd0_clc0.pkl'), "rb"))
        summary_token_lc = gen_tokens_lowercase(docstring_token)
        write_to_pickle(path, summary_token_lc, file_name='summary/cfp0_csi0_cfd0_clc1.pkl')

    # ---------summary_tokens filter punctuation------------
    cf.is_cfp = True
    if cf.is_cfp:
        path = cf.data_root_path
        summary_token_lc = pickle.load(open(os.path.join(path, 'summary/cfp0_csi0_cfd0_clc1.pkl'), "rb"))
        summary_token_filter = gen_code_tokens_filter_punctuation(summary_token_lc)
        write_to_pickle(path, summary_token_filter, file_name='summary/cfp1_csi0_cfd0_clc1.pkl')

    # ---------summary_tokens_split_identifier-------------
    cf.is_csi = True
    if cf.is_csi:
        path = cf.data_root_path
        summary_token_lc = pickle.load(open(os.path.join(path, 'summary/cfp1_csi0_cfd0_clc1.pkl'), "rb"))
        summary_token_split = gen_tokens_split_identifier(summary_token_lc)
        write_to_pickle(path, summary_token_split, file_name='summary/cfp1_csi1_cfd0_clc1.pkl')

    # ----------------------------code processing-------------------------------------
    print(90*"-")
    code_path = os.path.join(data_path, 'code')
    make_directory(code_path)

    # ---------code extraction------------
    cf.extract_code = True
    if cf.extract_code:
        print("start extract_code")
        start_time = time.perf_counter()
        # codes = {}
        # for part in parts:
        #     codes[part] = {}
        #     for fid in correct_rebuild_tree_fid[part]:
        # if cf.dataset_type == "codenet-v1":
        #     docstring_tokens = csn[part][fid]['function_tokens']
        #         # else:
        #         #     docstring_tokens = csn[part][fid]['code_tokens']
        #         docstring_tokens = csn[part][fid]['code_tokens']
        #         codes[part][fid] = docstring_tokens
        codes = {
            part: {fid: [token.value for token in list(javalang.tokenizer.tokenize(csn[part][fid]['code']))] for fid in
                   correct_rebuild_tree_fid[part]} for part in parts}
        write_to_pickle(code_path, codes, file_name='djl1_dfp0_dsi0_dlc0_dr0.pkl')
        print("extract code time cost: ", time.perf_counter() - start_time)

    # ---------code_tokens_replace <NUM> <STR>-------------
    cf.is_dr = True
    if cf.is_dr:
        path = cf.data_root_path
        code_tokens = pickle.load(open(os.path.join(path, 'code/djl1_dfp0_dsi0_dlc0_dr0.pkl'), "rb"))
        code_tokens_replace = gen_code_tokens_replace_str_num(code_tokens)
        write_to_pickle(path, code_tokens_replace, file_name='code/djl1_dfp0_dsi0_dlc0_dr1.pkl')
    # ---------code_tokens_split_identifier-------------
    cf.is_dsi = True
    if cf.is_dsi:
        path = cf.data_root_path
        code_tokens = pickle.load(open(os.path.join(path, 'code/djl1_dfp0_dsi0_dlc0_dr1.pkl'), "rb"))
        code_tokens_split = gen_tokens_split_identifier(code_tokens)
        write_to_pickle(path, code_tokens_split, file_name='code/djl1_dfp0_dsi1_dlc0_dr1.pkl')

    # ------------code_tokens lower case-------------
    cf.is_dlc = True
    if cf.is_dlc:
        path = cf.data_root_path
        code_tokens = pickle.load(open(os.path.join(path, 'code/djl1_dfp0_dsi1_dlc0_dr1.pkl'), "rb"))
        code_tokens_lc = gen_tokens_lowercase(code_tokens)
        write_to_pickle(path, code_tokens_lc, file_name='code/djl1_dfp0_dsi1_dlc1_dr1.pkl')

