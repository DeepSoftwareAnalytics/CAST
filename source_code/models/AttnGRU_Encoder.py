# !/usr/bin/env python
# !-*-coding:utf-8 -*- 
'''
@version: python3.7
@file: AttnGRU_Encoder.py
@time: 7/24/2020 9:11 PM
'''

import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from util.Config import Config as cf

class GRUEncoder(nn.Module):

    def __init__(self, vocab_size, emd_dim, hidden_size):
        super(GRUEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emd_dim)
        self.gru = nn.GRU(input_size=emd_dim, hidden_size=hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, input, hidden):
        embedded = self.embedding(input)  # [batch_size, seq_len, emd_dim]
        output, hidden = self.gru(embedded, hidden)
        # output [batch_size, seq_len, hidden_size]
        # hidden [num_layers * num_directions, batch_size, hidden_size]
        return output, hidden


class CodeTransEncoder(nn.Module):

    # def __init__(self, vocab_size, emd_dim, hidden_size):
    def __init__(self, vocab_size, emd_dim):
        super(CodeTransEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emd_dim)
        # self.gru = nn.GRU(input_size=emd_dim, hidden_size=hidden_size, batch_first=True)
        encoder_layers = TransformerEncoderLayer(d_model=cf.feature_dim, nhead=cf.nhead, dim_feedforward=cf.dim_feedforward, dropout=cf.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=cf.nlayers)

        # self.hidden_size = hidden_size

    # def forward(self, input, hidden):
    def forward(self, input):
        embedded = self.embedding(input)  # [batch_size, seq_len, emd_dim]
        # output, hidden = self.gru(embedded, hidden)
        max_len = cf.code_max_len
        batch_size = cf.batch_size
        embedded = embedded.view(max_len, batch_size, -1)  # (max_len, self.batch_size,feature_dim)
        memory = self.transformer_encoder(embedded)
        # memory (max_len, self.batch_size,feature_dim)
        memory_out = memory.view(batch_size, max_len, -1)  # (max_len, self.batch_size,feature_dim)

        # output [batch_size, seq_len, hidden_size]
        # hidden [num_layers * num_directions, batch_size, hidden_size]
        return  memory_out