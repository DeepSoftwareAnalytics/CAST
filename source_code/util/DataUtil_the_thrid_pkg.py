# !/usr/bin/env python
# !-*-coding:utf-8 -*- 

from multiprocessing import cpu_count, Pool
import re
from spiral import ronin
from util.DataUtil import array_split

# copy from CoCoGUM
def code_tokens_split_identifier(sequence):
    tokens = []
    for s in sequence:
        sub_sequence = [tok for tok in ronin.split(s) if tok]
        tokens.extend(sub_sequence)
    return tokens


# copy from CoCoGUM
def code_tokens_split_identifier_lists(seqs):
    return [code_tokens_split_identifier(seq) for seq in seqs]


# copy from CoCoGUM
def code_tokens_split_identifier_parallel(data):
    cores = cpu_count() - 1
    pool = Pool(cores)
    ids = list(data.keys())
    ids_split = array_split(ids, cores)
    data_split = []

    for split in ids_split:
        data_split.append([data[i] for i in split])

    results = pool.map(code_tokens_split_identifier_lists, data_split)
    new_data = {}
    for ids, result in zip(ids_split, results):
        for i in range(len(ids)):
            mid = ids[i]
            new_data[mid] = result[i]
    pool.close()
    pool.join()
    return new_data
