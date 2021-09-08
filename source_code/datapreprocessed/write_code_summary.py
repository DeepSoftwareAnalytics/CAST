#!/usr/bin/env python
# !-*-coding:utf-8 -*-
'''
@version: python3.*
@author: ‘v-ensh‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: get_code_summary.py
@time: 11/23/2020 5:24 PM
'''
import sys
import os

sys.path.append("../util")
from DataUtil import read_pickle_data, make_directory, time_format, delete_error_fids
import time
from Config import Config as cf


class Config:
    # csn_path = "../../../Data/codesearchnet_v1/"
    csn_path = cf.data_root_path
    # csn_code_path = "../../../Data/codesearchnet_v1/code"
    csn_code_path = os.path.join(cf.data_root_path, 'code')
    # csn_sum_path = "../../../Data/codesearchnet_v1/summary"
    csn_sum_path = os.path.join(cf.data_root_path, 'summary')
    csn_correct_fids_path = cf.correct_fid
    max_code_len = 170
    max_summary_len = 40
    truncation = False


def get_trunc_code_summary(part):
    trunc_code_token = [code_token[part][fid][:Config.max_code_len] for fid in correct_fid[part]]
    trunc_code_subtoken = [code_subtoken[part][fid][:Config.max_code_len] for fid in correct_fid[part]]
    trunc_sum_token = [summary_token[part][fid][:Config.max_summary_len] for fid in correct_fid[part]]
    return trunc_code_token, trunc_code_subtoken, trunc_sum_token


def get_code_summary(part):
    full_code_token = None
    if code_token:
        full_code_token = [code_token[part][fid] for fid in correct_fid[part]]
    full_code_subtoken = [code_subtoken[part][fid] for fid in correct_fid[part]]
    full_sum_token = [summary_token[part][fid] for fid in correct_fid[part]]
    return full_code_token, full_code_subtoken, full_sum_token


def load_data():
    code_token = read_pickle_data(os.path.join(Config.csn_code_path, "djl1_dfp0_dsi0_dlc0_dr0.pkl"))
    code_token = None
    code_subtoken = read_pickle_data(os.path.join(Config.csn_code_path, "djl1_dfp0_dsi1_dlc1_dr1.pkl"))
    summary_token = read_pickle_data(os.path.join(Config.csn_sum_path, "cfp1_csi1_cfd0_clc1.pkl"))
    correct_fid = read_pickle_data(os.path.join(Config.csn_correct_fids_path, "correct_rebuild_tree_fid.pkl"))

    # correct_fids = delete_error_fids(finally_summary_ids, tree_len_cnt_gt_1000_fid)
    return code_token, code_subtoken, summary_token, correct_fid


def write_to_file(path, filename, data):
    make_directory(path)
    with open(os.path.join(path, filename), "w") as f:
        for item in data:
            try:
                f.write(" ".join(item))
                f.write("\n")
            except UnicodeEncodeError as e:
                print(item)
                raise e
    print("wirite into %s " % os.path.join(path, filename))


def write_all_data_to_file(full_code_token, full_code_subtoken, full_sum_token, part="train"):

    write_to_file(os.path.join(Config.csn_path, "code_sum", part), "code.subtoken",
                  full_code_subtoken)
    write_to_file(os.path.join(Config.csn_path, "code_sum", part), "javadoc.original", full_sum_token)


if __name__ == '__main__':
    code_token, code_subtoken, summary_token, correct_fid = load_data()
    # make_dirs()

    # 1. train
    start = time.perf_counter()
    part = "train"
    if Config.truncation:
        trunc_code_token, trunc_code_subtoken, trunc_sum_token = get_trunc_code_summary(part)
        write_all_data_to_file(trunc_code_token, trunc_code_subtoken, trunc_sum_token, part="train")
        print("%s  examples :" % part + str(len(trunc_code_token)))
    else:
        full_code_token, full_code_subtoken, full_sum_token = get_code_summary(part)
        write_all_data_to_file(full_code_token, full_code_subtoken, full_sum_token, part="train")
        print("%s  examples :" % part + str(len(full_code_subtoken)))

    print("train  time cost :" + time_format(time.perf_counter() - start))

    # 2. dev
    st = time.perf_counter()
    part = "val"
    if Config.truncation:
        trunc_code_token, trunc_code_subtoken, trunc_sum_token = get_trunc_code_summary(part)
        write_all_data_to_file(trunc_code_token, trunc_code_subtoken, trunc_sum_token, part="val")
        print("%s  examples :" % part + str(len(trunc_code_token)))
    else:
        full_code_token, full_code_subtoken, full_sum_token = get_code_summary(part)
        write_all_data_to_file(full_code_token, full_code_subtoken, full_sum_token, part="val")
        print("%s  examples :" % part + str(len(full_code_subtoken)))
    print("val time cost :" + time_format(time.perf_counter() - st))

    # 3. test
    st = time.perf_counter()
    part = "test"
    if Config.truncation:
        trunc_code_token, trunc_code_subtoken, trunc_sum_token = get_trunc_code_summary(part)
        write_all_data_to_file(trunc_code_token, trunc_code_subtoken, trunc_sum_token, part="test")
        print("%s  examples :" % part + str(len(trunc_code_token)))
    else:
        full_code_token, full_code_subtoken, full_sum_token = get_code_summary(part)
        write_all_data_to_file(full_code_token, full_code_subtoken, full_sum_token, part="test")
        print("%s  examples :" % part + str(len(full_code_subtoken)))
    print("total time cost :" + time_format(time.perf_counter() - start))
