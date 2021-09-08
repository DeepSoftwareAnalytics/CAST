#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import sys
import os

sys.path.append("../util")
from DataUtil import read_pickle_data, make_directory, time_format, save_pickle_data
import time
from Config import Config as cf


class Config:
    csn_path = cf.data_root_path
    csn_ast_path = os.path.join(cf.data_root_path, 'ASTs')
    csn_correct_fids_path = cf.correct_fid
    use_codebert_data = False
    use_subtoken_dataset = True


def load_data():
    ast_token = read_pickle_data(os.path.join(Config.csn_ast_path, "sliced_AST_subtoken.pkl"))
    correct_fid = read_pickle_data(os.path.join(Config.csn_correct_fids_path, "correct_rebuild_tree_fid.pkl"))

    return ast_token, correct_fid


if __name__ == '__main__':
    splitted_ast, correct_fid = load_data()
    start = time.perf_counter()
    for part in splitted_ast:
        st = time.perf_counter()
        splitted_ast_part = {fid: splitted_ast[part][fid] for fid in correct_fid[part]}

        save_pickle_data(os.path.join(Config.csn_ast_path, part), 'split_AST.pkl', splitted_ast_part)
        print("%s time cost :" % part + time_format(time.perf_counter() - st))
        print("%s examples :" % part + str(len(splitted_ast_part)))
    print("total time cost :" + time_format(time.perf_counter() - start))
