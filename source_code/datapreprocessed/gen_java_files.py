#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import time
import sys
import pickle
import os

sys.path.append("..")
sys.path.append("../util")
from Config import Config as cf
from DataUtil import make_directory, update_dict, time_format
from LoggerUtil import set_logger, debug_logger


def select_dict_value(data, col):
    res_dict = {}
    for k, v in data.items():
        res_dict[k] = v[col]
    return res_dict


def get_source_code(data):
    """
    Generate code_dict = {k:{fid:code}}, k in ['train', 'val', 'test'].
    :param data:
    :return:
    """
    code_dict = {}
    for partition in data.keys():
        code_dict[partition] = select_dict_value(data[partition], 'code')
        for k, v in code_dict[partition].items():
            code_dict[partition][k] = v + '\n'
    return code_dict


def add_class_name(data):
    code_dict = {}
    for partition in data.keys():
        code_dict[partition] = {fid: "public class Demo_%d {\n\t%s\n}" % (fid, method) for fid, method in
                                data[partition].items()}
    return code_dict


def write_code_to_java_file(method):
    """
    write code to a java file
    :param method: (fid, code_string)
    :return:
    """
    fid, method_code = method[0], method[1]
    java_path = os.path.join(cf.java_files_dir, str(fid) + ".java")
    with open(java_path, "w") as f:
        f.write(method_code)
    return java_path


def add_key(d):
    """
    Take a dictionary d {k:v}, return {k:(k,v)}
    """
    return {k: (k, v) for k, v in d.items()}


def merge_train_val_test(data):
    new_data = {}
    for part in data:
        new_data.update(data[part])
    return new_data


def dict_slice(adict, start, end):
    # return {k: adict[k] for k in list(adict.keys())[start:end]}
    return [(k, adict[k]) for k in list(adict.keys())[start:end]]


if __name__ == '__main__':
    log_file ="./log/1_gen_java_files.txt"
    print('1')
    set_logger(cf.debug, log_file, checkpoint=True)
    start = time.perf_counter()
    csn = pickle.load(open(os.path.join(cf.data_root_path, '../../Data/TL_CodeSum/csn.pkl'), "rb"))
    #
    source_code_dict = get_source_code(csn)
    source_code_dict = add_class_name(source_code_dict)

    debug_logger("write_all_data_into_several_folders")
    # cf.java_files_dir = os.path.join(cf.data_root_path, 'java_files')
    cf.java_files_dir = os.path.join("../../Data/TL_CodeSum", 'java_files')
    debug_logger("write_all_data_into_%s" %cf.java_files_dir)
    make_directory(cf.java_files_dir)
    [write_code_to_java_file(item) for partition in source_code_dict for item in source_code_dict[partition].items()]
    debug_logger(" time cost :" + time_format(time.perf_counter() - start))
