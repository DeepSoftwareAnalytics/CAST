# calculate cliffs delta effect size: https://github.com/neilernst/cliffsDelta/blob/master/cliffsDelta.py
from __future__ import division
import copy
import random
from collections import Counter
import numpy as np
from prettytable import PrettyTable
import os


def cliffsDelta(lst1, lst2, **dull):
    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474}  # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j * repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j) * repeats
    d = (more - less) / (m * n)
    size = lookup_size(d, dull)
    return d, size


def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull['small']:
        return 'negligible'
    if dull['small'] <= delta < dull['medium']:
        return 'small'
    if dull['medium'] <= delta < dull['large']:
        return 'medium'
    if delta >= dull['large']:
        return 'large'


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two


### calculate wilcoxon pvalue
import pandas as pd
from scipy import stats


def wilcoxon_signed_rank_test(y1, y2):
    statistic, pvalue = stats.wilcoxon(y1, y2)
    return pvalue


def get_score(y1, y2):
    pvalue = wilcoxon_signed_rank_test(y1, y2)
    d, size = cliffsDelta(y1, y2)
    return pvalue, d, size


def get_pvalue_and_effect_size(all_score):
    models_name = list(all_score)
    for i in range(len(models_name)):
        for j in range(i + 1, len(models_name)):
            pvalue, d, size = get_score(all_score[models_name[i]], all_score[models_name[j]])
            print(
                "{} and {}, pvalue:{}, cliffsDelta:{}, effect size:{}".format(models_name[i], models_name[j], pvalue, d,
                                                                              size))


def get_all_model_score(path, question_cnt=50):
    data_frame = pd.read_excel(path)
    result = {}

    user_cnt = len(data_frame["ID"])
    for i in range(user_cnt):
        for q in range(question_cnt):
            cocogum, ast_att_gru, astnn, rencos = list(data_frame.loc[i])[5 + 4 * q:9 + 4 * q]
            key = "Q_" + str(q)
            if key in result:
                result[key]["model4"].append(cocogum)
                result[key]["model1"].append(ast_att_gru)
                result[key]["model2"].append(astnn)
                result[key]["model3"].append(rencos)
            else:
                result[key] = {}
                result[key]["model4"] = [cocogum]
                result[key]["model1"] = [ast_att_gru]
                result[key]["model2"] = [astnn]
                result[key]["model3"] = [rencos]

    cocogum_scores = []
    ast_att_gru_scores = []
    astnn_scores = []
    rencos_scores = []
    for q, four_score in result.items():
        cocogum_scores.extend(four_score["model4"])
        ast_att_gru_scores.extend(four_score["model1"])
        astnn_scores.extend(four_score["model2"])
        rencos_scores.extend(four_score["model3"])

    all_score = {"model4": cocogum_scores, "model1": ast_att_gru_scores, "model2": astnn_scores,
                 "model3": rencos_scores}
    return all_score


def parse_score_dict(result):
    cocogum_scores = []
    ast_att_gru_scores = []
    astnn_scores = []
    rencos_scores = []
    for q, four_score in result.items():
        cocogum_scores.extend(four_score["model4"])
        ast_att_gru_scores.extend(four_score["model1"])
        astnn_scores.extend(four_score["model2"])
        rencos_scores.extend(four_score["model3"])

    all_score = {"model4": cocogum_scores, "model1": ast_att_gru_scores, "model2": astnn_scores,
                 "model3": rencos_scores}
    return all_score


def get_all_model_in_three_aspects_score(path, question_cnt=10, start_qid=1):
    model_order_dict = {}
    for i in range(0, 50, 1):
        random.seed(i)
        li = [1, 2, 3, 4]
        random.shuffle(li)
        model_order_dict[i] = li
    #     print('model_order_dict:', model_order_dict)

    #     path = "112506357_2_Code Summarization Human Evaluation 1- 10_2_2.xlsx"
    data_frame = pd.read_excel(path)
    result = {"informative": {}, "naturalness": {}, "similarity": {}}
    user_cnt = len(data_frame["序号"])
    for i in range(user_cnt):
        for q in range(question_cnt):
            start_index = 6 + q * 12
            one_question_score = [list(data_frame.loc[i])[start_index + j * 3:start_index + (j + 1) * 3] for j in
                                  range(4)]
            model_order_in_this_question = model_order_dict[q + start_qid  ]
            one_question_model_socre = dict(zip(model_order_in_this_question, one_question_score))
            model1, model2, model3, model4 = one_question_model_socre[1], \
                                                  one_question_model_socre[2], \
                                                  one_question_model_socre[3], \
                                                  one_question_model_socre[4]
            key = "Q_" + str(q)
            if key in result["informative"]:
                result["informative"][key]["model4"].append(model4[0] - 1)
                result["informative"][key]["model1"].append(model1[0] - 1)
                result["informative"][key]["model2"].append(model2[0] - 1)
                result["informative"][key]["model3"].append(model3[0] - 1)

                result["naturalness"][key]["model4"].append(model4[1] - 1)
                result["naturalness"][key]["model1"].append(model1[1] - 1)
                result["naturalness"][key]["model2"].append(model2[1] - 1)
                result["naturalness"][key]["model3"].append(model3[1] - 1)

                result["similarity"][key]["model4"].append(model4[2] - 1)
                result["similarity"][key]["model1"].append(model1[2] - 1)
                result["similarity"][key]["model2"].append(model2[2] - 1)
                result["similarity"][key]["model3"].append(model3[2] - 1)
            else:
                result["informative"][key] = {}
                result["naturalness"][key] = {}
                result["similarity"][key] = {}

                result["informative"][key]["model4"] = [model4[0] - 1]
                result["informative"][key]["model1"] = [model1[0] - 1]
                result["informative"][key]["model2"] = [model2[0] - 1]
                result["informative"][key]["model3"] = [model3[0] - 1]

                result["naturalness"][key]["model4"] = [model4[1] - 1]
                result["naturalness"][key]["model1"] = [model1[1] - 1 - 1]
                result["naturalness"][key]["model2"] = [model2[1] - 1]
                result["naturalness"][key]["model3"] = [model3[1] - 1]

                result["similarity"][key]["model4"] = [model4[2] - 1]
                result["similarity"][key]["model1"] = [model1[2] - 1]
                result["similarity"][key]["model2"] = [model2[2] - 1]
                result["similarity"][key]["model3"] = [model3[2] - 1]
    return parse_score_dict(result['informative']), parse_score_dict(result['naturalness']), parse_score_dict(
        result['similarity'])


def print_distribution(four_model_score):
    table = PrettyTable(['model type', "0", "1", "2", "3", "4", "Avg(Std)", "≥3", "≥2", "≤1"])
    for k in four_model_score:
        result = Counter(four_model_score[k])
        avg = np.mean(four_model_score[k])
        std = np.std(four_model_score[k])
        table.add_row([k, result[0], result[1], result[2], result[3], result[4],
                       "{}({})".format(round(avg, 2), round(std, 2)),
                       result[3] + result[4], result[2] + result[3] + result[4], result[0] + result[1]])
    print(table)


# multi-excel
def merge_all_score(s1, s2, s3, s4, s5):
    merged_scores = copy.deepcopy(s1)
    five_score = [s1, s2, s3, s4, s5]
    for key in merged_scores[0].keys():
        for i in range(3):
            for j in range(1, len(five_score)):
                merged_scores[i][key].extend(five_score[j][i][key])
    return merged_scores


def calcute_final_result(path1, path2, path3, path4, path5, model_dict):
    all_scores1_10 = get_all_model_in_three_aspects_score(path1, question_cnt=10, start_qid=0)
    all_scores11_20 = get_all_model_in_three_aspects_score(path2, question_cnt=10, start_qid=10)
    all_scores21_30 = get_all_model_in_three_aspects_score(path3, question_cnt=10, start_qid=20)
    all_scores31_40 = get_all_model_in_three_aspects_score(path4, question_cnt=10, start_qid=30)
    all_scores41_50 = get_all_model_in_three_aspects_score(path5, question_cnt=10, start_qid=40)
    merged_scores = merge_all_score(all_scores1_10, all_scores11_20, all_scores21_30, all_scores31_40, all_scores41_50)
    print("informative")
    result = {model_dict[key]: value for key, value in merged_scores[0].items()}
    print_distribution(result)
    get_pvalue_and_effect_size(result)

    print(80 * "*")
    print("naturalness")
    result = {model_dict[key]: value for key, value in merged_scores[1].items()}
    print_distribution(result)
    get_pvalue_and_effect_size(result)

    print(80 * "*")
    print("similarity")
    result = {model_dict[key]: value for key, value in merged_scores[2].items()}
    print_distribution(result)

    get_pvalue_and_effect_size(result)
    return merged_scores


def main():
    response_dir = "response"
    path1_10 = os.path.join(response_dir, r"117029055_2_Code Summarization Human Evaluation 1- 10 (EMNLP)_4_4.xlsx")
    path11_20 = os.path.join(response_dir, r"117025035_2_Code Summarization Human Evaluation 11- 20  (EMNLP)_4_4.xlsx")
    path21_30 = os.path.join(response_dir, r"117023218_2_Code Summarization Human Evaluation 21- 30  (EMNLP)_4_4.xlsx")
    path31_40 = os.path.join(response_dir, r"117038567_2_Code Summarization Human Evaluation 31- 40  (EMNLP)_4_4.xlsx")
    path41_50 = os.path.join(response_dir, r"117037430_2_Code Summarization Human Evaluation 41- 50  (EMNLP)_4_4.xlsx")
    model_dict = {"model1": "Ast-attendgru", "model2": "NCS", "model3": "CodeASTNN", "model4": "CAST"}
    scores = calcute_final_result(path1_10, path11_20, path21_30, path31_40, path41_50, model_dict)


if __name__ == "__main__":
    """ run the following command:
    python human_eval.py 2>&1 | tee human_eval_result.log
    """
    main()
