#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import time
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
import os
import sys
import pickle
from multiprocessing import cpu_count, Pool

sys.path.append("../util")
from Config import Config as cf
from LoggerUtil import set_logger, debug_logger
from DataUtil import save_pickle_data, time_format


def is_correct_graph(data):
    if type(data) == dict:
        id = list(data.keys)[0]
    else:
        id = data
    dot_file_log = os.path.join(cf.dot_files_dir, "%s.java.ast.err.log" % str(id))
    if os.path.getsize(dot_file_log):
        return False
    else:
        return True


class Config:
    is_multi_processing = True
    is_full_ast = False


if __name__ == '__main__':

    log_file = "./log/3_get_correct_splitted_ast.txt"
    set_logger(cf.debug, log_file, checkpoint=True)
    start = time.perf_counter()
    csn = pickle.load(open(os.path.join(cf.data_root_path, '../../Data/TL_CodeSum/csn.pkl'), "rb"))


    debug_logger("Start count correct fid")
    if Config.is_multi_processing:
        cores = cpu_count()
        correct_splitted_ast_fid = {}
        for part in ["val", "train", "test"]:
            pool = Pool(cores)
            results = pool.map(is_correct_graph, csn[part])
            pool.close()
            pool.join()
            correct_splitted_ast_fid[part] = [fid for i, fid in enumerate(csn[part]) if results[i]]
            debug_logger("%s Done" % part)
    else:
        correct_splitted_ast_fid = {part: [fid for fid in csn[part] if is_correct_graph(fid)] for part in csn}
    if Config.is_full_ast:
        save_pickle_data(os.path.join(cf.correct_fid, "full_ast"), "correct_splitted_ast_fid.pkl", correct_splitted_ast_fid)

    else:
        save_pickle_data(cf.correct_fid, "correct_splitted_ast_fid.pkl", correct_splitted_ast_fid)
    for part in correct_splitted_ast_fid:
        debug_logger(" %s length: %d " % (part, len(correct_splitted_ast_fid[part])))
    debug_logger(" time cost :" + time_format(time.perf_counter() - start))
