# !/usr/bin/env python
# !-*-coding:utf-8 -*- 
'''
@version: python3.7
@file: RvNNEncoder.py
@time: 7/9/2020 9:25 AM
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from util.Config import Config as cf
from models.WeightedRvNN import WeightedBatchSubTreeEncoder


#  copy from https://github.com/zhangj111/astnn/blob/master/model.py
class BatchSubTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchSubTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        # self.W_l = nn.Linear(encode_dim, encode_dim)
        # self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            try:
                if node[i][0] is not -1:
                    index.append(i)
                    current_node.append(node[i][0])
                    temp = node[i][1:]
                    c_num = len(temp)
                    for j in range(c_num):
                        if temp[j][0] is not -1:
                            if len(children_index) <= j:
                                children_index.append([i])
                                children.append([temp[j]])
                            else:
                                children_index[j].append(i)
                                children[j].append(temp[j])
                else:
                    batch_index[i] = -1
            except:
                pass

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        try:
            self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        except:
            pass
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        if cf.node_combine == "max":
            return torch.max(self.node_list, 0)[0]
        elif cf.node_combine == "mean":
            return torch.mean(self.node_list, 0)
        # return torch.max(self.node_list, 0)[0]


#  revise from https://github.com/zhangj111/astnn/blob/master/model.py
class BatchASTEncoder(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, batch_size, use_gpu=True,
                 pretrained_weight=None):
        super(BatchASTEncoder, self).__init__()
        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        # self.gpu = use_gpu
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        # self.label_size = label_size
        # class "BatchTreeEncoder"
        if cf.is_weighted_RvNN:
            self.encoder = WeightedBatchSubTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                                       self.batch_size, self.gpu, pretrained_weight)
        else:
            self.encoder = BatchSubTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                               self.batch_size, self.gpu, pretrained_weight)
        # self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        # self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            try:
                local_rank = torch.distributed.get_rank()
            except AssertionError:
                local_rank = 0
            device = torch.device("cuda", local_rank)
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0.to(device), c0.to(device)
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda().to(device)
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

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
        gru_out, hidden = self.bigru(encodes, self.hidden)

        # gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        # gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]

        # linear
        # y = self.hidden2label(gru_out)
        return gru_out, hidden
        # gru_out [1,3,20] [batch_size,seq_len,2*hidden_dim]
        # hidden [2,1,10] [batch_size,seq_len,2*hidden_dim]


#  revise from https://github.com/zhangj111/astnn/blob/master/model.py
class BatchASTRvNNEncoder(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    def __init__(self, embedding_dim, vocab_size, encode_dim, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchASTRvNNEncoder, self).__init__()
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        if cf.is_weighted_RvNN:
            self.encoder = WeightedBatchSubTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                                       self.batch_size, self.gpu, pretrained_weight)
        else:
            self.encoder = BatchSubTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                               self.batch_size, self.gpu, pretrained_weight)

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

        return encodes
