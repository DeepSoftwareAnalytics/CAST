# !/usr/bin/env python
# !-*-coding:utf-8 -*- 

import nltk
# import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from torch.utils.data.dataloader import DataLoader
# import sys
# # sys.path.append("./")
from util.Config import Config as cf
from util.CustomedBleu.bleu import _bleu
from util.CustomedBleu.smooth_bleu import smooth_bleu
from util.DataUtil import read_pickle_data,stem_and_replace_predict_target
# from util.Dataset import CodeSummaryDataset
# from util.GPUUtil import move_to_device, move_pyg_to_device
# from torch_geometric.data import Batch
# # from rouge import Rouge
from util.meteor.meteor import Meteor
from util.rouge.rouge import Rouge
from util.cider.cider import Cider
import math


def bleu_so_far(refs, preds):
    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue

        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    try:
        bleu_list = smooth_bleu(r_str_list, p_str_list)
    except:
         bleu_list = [0, 0, 0, 0]

    return "_", "_", bleu_list[0]


def metetor_rouge_cider(refs, preds):
    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i][0])]

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    # print("ROUGe: ", score_Rouge)

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    # print("Cider: ", score_Cider)

    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    # print("Meteor: ", score_Meteor)
    return score_Rouge, score_Cider,  score_Meteor

