#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import json

import argparse
import os
import time
from util.DataUtil import time_format
from util.EvaluateUtil import bleu_so_far, metetor_rouge_cider
if __name__ == '__main__':
    s_time = time.perf_counter()


    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default="Funcom", required=False)
    args_cmd = parser.parse_args()
    path = "../output/%s/predict.json" % args_cmd.dataset

    if not os.path.exists(path):
        raise FileExistsError("%s is not exit" % path)

    with open(path, "r") as f:
        data = f.readlines()

    data = [json.loads(item) for item in data]
    preds, refs = zip(*[(item["predictions"][0].split(), [item['references'][0].split()]) for item in data])
    ret_val_full, _, bleu_val_full = bleu_so_far(refs, preds)
    score_Rouge, score_Cider, score_Meteor = metetor_rouge_cider(refs, preds)
    print('Bleu = %.2f\nMeteor = %.2f \nRouge = %.2f \nCider = %.2f ' % (
    bleu_val_full, score_Meteor*100, score_Rouge*100, score_Cider))
    print("time cost: %s" %
                time_format(time.perf_counter() - s_time))
