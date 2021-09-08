# !/usr/bin/env python
# !-*-coding:utf-8 -*- 

import torch
from torch.utils.data import Dataset
import torch
import re
from torch._six import container_abcs, string_classes, int_classes
from tqdm import tqdm
from copy import deepcopy
# from util.treelstm.tree import Tree
import os


class CodeSummaryDataset(Dataset):
    def __init__(self, summary, asts, code=None):
        super(CodeSummaryDataset, self).__init__()

        self.seq_num = len(summary)

        assert (len(asts) == self.seq_num)

        self.summary = []
        self.asts = []

        if code is not None:
            assert (len(code) == self.seq_num)
            self.code = []
        else:
            self.code = None

        for key, seq in summary.items():
            self.summary.append(seq)
            self.asts.append(asts[key])
            if code is not None:
                self.code.append(code[key])

        # Use only when summary, code and sbt are padded already
        # and each sequence has the same length
        self.summary = torch.LongTensor(self.summary)
        # self.asts = torch.LongTensor(self.asts)

        if self.code is not None:
            self.code = torch.LongTensor(self.code)

    def __len__(self):
        return self.seq_num

    def __getitem__(self, idx):

        if self.code is not None:
            return self.asts[idx], self.code[idx], self.summary[idx]
        else:
            return self.asts[idx], self.summary[idx]


def collate_function(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    bacth_size = len(batch)
    item_len = len(elem)
    summary_len = len(elem[-1])
    code_len = len(elem[1])
    if item_len == 2:
        item1 = []
        item2 = torch.zeros(bacth_size, summary_len)
        for i in range(bacth_size):
            item1.append(batch[i][0])
            item2[i, :] = batch[i][1]
        return [item1, item2.long()]
    if item_len == 3:
        item1 = []
        item2 = torch.zeros(bacth_size, code_len)
        item3 = torch.zeros(bacth_size, summary_len)

        for i in range(bacth_size):
            item1.append(batch[i][0])
            item2[i, :] = batch[i][1]
            item3[i, :] = batch[i][-1]
        return [item1, item2.long(), item3.long()]



class SplittedASTDataset(Dataset):
    def __init__(self, summary, asts, rebuild_tree, code=None):
        super(SplittedASTDataset, self).__init__()

        self.seq_num = len(summary)

        assert (len(asts) == self.seq_num)

        self.summary = []
        self.asts = []
        self.rebuild_tree = []

        if code is not None:
            assert (len(code) == self.seq_num)
            self.code = []
        else:
            self.code = None

        for key, seq in summary.items():
            self.summary.append(seq)
            self.asts.append(asts[key])
            self.rebuild_tree.append(rebuild_tree[key])
            if code is not None:
                self.code.append(code[key])

        # Use only when summary, code and sbt are padded already
        # and each sequence has the same length
        self.summary = torch.LongTensor(self.summary)
        # self.asts = torch.LongTensor(self.asts)

        if self.code is not None:
            self.code = torch.LongTensor(self.code)

    def __len__(self):
        return self.seq_num

    def __getitem__(self, idx):

        if self.code is not None:
            return self.asts[idx], self.rebuild_tree[idx], self.code[idx], self.summary[idx]
        else:
            return self.asts[idx], self.rebuild_tree[idx], self.summary[idx]


def collate_function_splitted_AST(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    bacth_size = len(batch)
    item_len = len(elem)
    summary_len = len(elem[-1])
    code_len = len(elem[-2])
    if item_len == 3:
        item1 = []
        item2 = []
        item3 = torch.zeros(bacth_size, summary_len)
        for i in range(bacth_size):
            item1.append(batch[i][0])
            item2.append(batch[i][1])
            item3[i, :] = batch[i][-1]
        return [item1, item2, item3.long()]
    if item_len == 4:
        item1 = []
        item2 = []
        item3 = torch.zeros(bacth_size, code_len)
        item4 = torch.zeros(bacth_size, summary_len)

        for i in range(bacth_size):
            item1.append(batch[i][0])
            item2.append(batch[i][1])
            item3[i, :] = batch[i][-2]
            item4[i, :] = batch[i][-1]
        return [item1, item2, item3.long(), item4.long()]
