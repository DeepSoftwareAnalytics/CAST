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
import sys

#  copy from https://github.com/zhangj111/astnn/blob/master/model.py
class WeightedBatchSubTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(WeightedBatchSubTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_sum = nn.Linear(encode_dim, encode_dim)
        # self.W_l = nn.Linear(encode_dim, encode_dim)
        # self.W_r = nn.Linear(encode_dim, encode_dim)
        if cf.activate_f == "relu":
            self.activation = F.relu
        elif cf.activate_f == "tanh":
            self.activation = F.tanh
        # self.activation = torch.tanh

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
        children_weight = []
        for i in range(size):
            try:
                if node[i][0] is not -1:
                    index.append(i)
                    current_node.append(node[i][0])
                    temp = node[i][1:]
                    c_num = len(temp)
                    c_weight = 0 if c_num == 0 else 1 / c_num
                    children_weight.append(c_weight)
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

        # batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
        #                                                   self.embedding(Variable(self.th.LongTensor(current_node)))))
        batch_current = batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                 self.W_c(self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                if cf.is_avg_weighted_RvNN and max(children_weight) > 1:
                    # zeros_w = self.create_tensor(Variable(torch.zeros(size)))
                    # c_w = (torch.zeros(size).index_copy(0, Variable(self.th.LongTensor(children_index[c])),
                    #                                     torch.tensor(children_weight))).reshape((size, 1))
                    c_w = torch.tensor([[children_weight[i] if i in children_index[c] else 0] for i in range(size)])
                    # batch_current += 1/len(children) * self.W_sum(zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree))
                    batch_current += self.W_sum(
                        zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)) * c_w
                else:
                    # try:
                    batch_current += self.W_sum(
                    zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree))
                    # except RuntimeError:
                    #     print(len(self.node_list))
                    #     print(sys.getsizeof(tree))
                    #     print(sys.getsizeof(batch_current))
                    #     raise RuntimeError

        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        try:
            self.node_list.append(self.activation(self.batch_node.index_copy(0, b_in, batch_current)))
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


#  copy from https://github.com/zhangj111/astnn/blob/master/model.py
class WeightedBatchTreeEncoder(nn.Module):
    def __init__(self, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(WeightedBatchTreeEncoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = None
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_sum = nn.Linear(encode_dim, encode_dim)
        # self.W_l = nn.Linear(encode_dim, encode_dim)
        # self.W_r = nn.Linear(encode_dim, encode_dim)
        # self.activation = F.relu
        # self.activation = torch.tanh
        if cf.activate_f == "relu":
            self.activation = F.relu
        elif cf.activate_f == "tanh":
            self.activation = F.tanh

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
        children_weight = []
        for i in range(size):
            try:
                if node[i][0] is not -1:
                    index.append(i)
                    current_node.append(node[i][0])
                    temp = node[i][1:]
                    c_num = len(temp)
                    c_weight = 0 if c_num == 0 else 1 / c_num
                    children_weight.append(c_weight)
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
        # self.embedding(Variable(self.th.LongTensor(current_node)))
        # node_embed = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        # node_embed.index_select(0, Variable(self.th.LongTensor(current_node)), self.embedding)
        # batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)), node_embed))
        batch_current = batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                 self.W_c(self.embedding(Variable(self.th.LongTensor(current_node)))))
        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                if cf.is_avg_weighted_RvNN and max(children_weight) > 1:
                    c_w = torch.tensor([[children_weight[i] if i in children_index[c] else 0] for i in range(size)])
                    batch_current += self.W_sum(
                        zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)) * c_w
                else:
                    batch_current += self.W_sum(
                        zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree))
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        try:
            self.node_list.append(self.activation(self.batch_node.index_copy(0, b_in, batch_current)))
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
