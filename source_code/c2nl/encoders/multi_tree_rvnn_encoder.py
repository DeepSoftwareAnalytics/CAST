#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import sys

sys.path.append("../../..")
# from source_code.models.RvNNEncoder import BatchASTRvNNEncoder


from source_code.models.WeightedRvNN import WeightedBatchSubTreeEncoder


#  revise from https://github.com/zhangj111/astnn/blob/master/model.py
class MultiTreeRvNNEncoder(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, batch_size, use_gpu=True,
    #              pretrained_weight=None):
    def __init__(self, args,
                 pretrained_weight=None):
        super(MultiTreeRvNNEncoder, self).__init__()
        # self.hidden_dim = args.RvNN_hidden_dim
        self.gpu = args.use_gpu
        self.batch_size = args.batch_size
        self.vocab_size = args.ast_vocab_size
        self.embedding_dim = args.node_embedding_dim
        self.encode_dim = args.RvNN_input_dim
        self.encoder = WeightedBatchSubTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                                   self.batch_size, self.gpu, pretrained_weight)
        # self.encoder = BatchASTRvNNEncoder(self.embedding_dim, self.vocab_size, self.encode_dim, self.batch_size,
        #                                    self.gpu, pretrained_weight)

        self.dropout = nn.Dropout(0.2)

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            try:
                for j in range(lens[i]):
                    encodes.append(x[i][j])
            except:
                pass

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len - lens[i]:
                seq.append(self.get_zeros(max_len - lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)

        # gru
        # gru_out, hidden = self.bigru(encodes, self.hidden)
        # gru_out = encodes
        # gru_out = torch.transpose(gru_out, 1, 2)

        return encodes
        # gru_out [1,3,20] [batch_size,seq_len,2*hidden_dim]
        # hidden [2,1,10] [batch_size,seq_len,2*hidden_dim]
