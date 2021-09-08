#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import sys
import os
import time

sys.path.append('../util')
from Config import Config as cf
from DataUtil import read_pickle_data, save_pickle_data, time_format


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False
    else:
        return True


def correct_summary_fids(data):
    gt3_fids = [fid for fid, item in data.items() if len(item) > 3]
    english_fids = [fid for fid, item in data.items() if isEnglish(" ".join(item))]
    return list(set(gt3_fids).intersection(set(english_fids)))


if __name__ == '__main__':
    start_time = time.perf_counter()
    summary = read_pickle_data(os.path.join(cf.data_root_path, 'summary/cfp1_csi1_cfd0_clc1.pkl'))
    finally_summary_id = {}
    for part in summary:
        summary_id = correct_summary_fids(summary[part])
        finally_summary_id[part] = summary_id
    save_pickle_data(cf.correct_fid, "finally_summary_id.pkl", finally_summary_id)
    for part in finally_summary_id:
        print(" %s length: %d " % (part, len(finally_summary_id[part])))
    print("time cost %s" % time_format(time.perf_counter() - start_time))
