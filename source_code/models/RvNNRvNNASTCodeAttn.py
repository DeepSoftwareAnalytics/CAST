# !/usr/bin/env python
# !-*-coding:utf-8 -*- 
'''
@version: python3.7
@file: RvNNBiGRU_GRU.py
@time: 7/9/2020 9:53 AM
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.RvNNEncoder import BatchASTEncoder
from models.AttnGRUDecoder import AttGRUDecoder, CodeAttnGRUDecoder
from models.AttnGRU_Encoder import GRUEncoder

from util.LoggerUtil import debug_logger
from util.GPUUtil import move_to_device
from util.Config import Config as cf
from torch.autograd import Variable
from models.WeightedRvNN import WeightedBatchSubTreeEncoder


def convert_rebuild_tree_to_RvNN_format(root_node, tree, tree_with_label, start_index):
    #     print("sequence start",sequence)
    children = tree[root_node]
    for i, r in enumerate(children):
        tree_with_label.append([r + start_index])
        if r in tree.keys():
            convert_rebuild_tree_to_RvNN_format(r, tree, tree_with_label[-1], start_index)
        else:
            continue


class BatchASTEncoder(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, batch_size, use_gpu=True,
                 pretrained_weight=None):
        super(BatchASTEncoder, self).__init__()

        # self.hidden_dim = hidden_dim
        # self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.encode_dim2 = cf.aggregate_dim

        if cf.is_weighted_RvNN:
            self.sub_tree_encoder = WeightedBatchSubTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                                                self.batch_size, self.gpu, pretrained_weight)
            self.rebuild_tree_encoder = WeightedBatchTreeEncoder(self.encode_dim, self.encode_dim2,
                                                                 self.batch_size, self.gpu, pretrained_weight)

        else:
            raise Exception("please use weighted_RvNN instead")

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x, rebuild_x):
        lens = [len(item) for item in x]
        # max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            encodes.extend(x[i])


        encodes = self.sub_tree_encoder(encodes, sum(lens))
        rebuild_trees = []
        start_index = 0
        for i in range(self.batch_size):
            tree = rebuild_x[i]
            root_node = 0
            tree_with_label = [0 + start_index]
            convert_rebuild_tree_to_RvNN_format(root_node, tree, tree_with_label, start_index)
            start_index += lens[i]
            rebuild_trees.append(tree_with_label)

        self.rebuild_tree_encoder.embedding = encodes
        # try:
        rebuild_trees_out = self.rebuild_tree_encoder(rebuild_trees, self.batch_size)
        # except RuntimeError as e:
        #     print("rebuild_x", rebuild_x)
        #     print("lens", lens)
        #     print(" x", x[-1])
        # rebuild_trees_out [batch_size, dim]
        # seq, start, end = [], 0, 0

        return rebuild_trees_out
        # return gru_out, hidden
        # gru_out [1,3,20] [batch_size,seq_len,2*hidden_dim]
        # hidden [2,1,10] [batch_size,seq_len,2*hidden_dim]


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
        # self.embedding(Variable(self.th.LongTensor(current_node)))
        # node_embed = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        # node_embed.index_select(0, Variable(self.th.LongTensor(current_node)), self.embedding)
        # batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)), node_embed))
        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding.index_select(0, Variable(
                                                              self.th.LongTensor(current_node)))))
        # batch_current = batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
        #                                          self.W_c(self.embedding(Variable(self.th.LongTensor(current_node)))))
        for c in range(len(children)):
            # try:
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            # except RuntimeError as e:
            #     print("size",size, self.encode_dim)
            #     raise e
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
        # try:
        self.traverse_mul(x, list(range(self.batch_size)))
        # except RuntimeError as e:
        #     print(self.embedding.shape)
        #     print("x", x)

            # raise e
        self.node_list = torch.stack(self.node_list)
        # [seq_len, batch_size, dim]
        if cf.node_combine == "max":
            return self.node_list, torch.max(self.node_list, 0)[0]

        # [seq_len, batch_size, dim] -->[batch_size, dim]
        elif cf.node_combine == "mean":
            return self.node_list, torch.mean(self.node_list, 0)
        # return torch.max(self.node_list, 0)[0]


class RvNNRvNNASTCodeAttn(nn.Module):

    def __init__(self, ):
        super(RvNNRvNNASTCodeAttn, self).__init__()
        self.batch_size = cf.batch_size
        self.asts_encoder = BatchASTEncoder(embedding_dim=cf.node_embedding_dim, hidden_dim=cf.bigru_hidden_dim,
                                            vocab_size=cf.ast_vocab_size, encode_dim=cf.RvNN_input_dim,
                                            batch_size=self.batch_size,
                                            use_gpu=cf.use_gpu, pretrained_weight=cf.pretrained_weight)

        self.code_encoder = GRUEncoder(vocab_size=cf.code_vocab_size, emd_dim=cf.code_embedding_dim,
                                       hidden_size=cf.code_gru_hidden_size)

        self.decoder = CodeAttnGRUDecoder(hidden_size=cf.decoder_rnn_hidden_size,
                                          vocab_size=cf.summary_vocab_size, emd_dim=cf.summary_embedding_dim)

    def forward(self, slicedAST, rebuild_tree, summary, method_code, use_teacher=True):

        asts_encoder_output, asts_encoder_hidden = self.asts_encoder(slicedAST, rebuild_tree)
        # asts_encoder_output [seq_len, batch_size, hidden_size]  node embedding
        # asts_encoder_hidden [ batchsize, hidden_size]  tree embedding

        asts_encoder_output = asts_encoder_output.permute(1, 0, 2)
        # asts_encoder_output [batch_size, seq_len, hidden_size]
        code_encoder_hidden = asts_encoder_hidden.unsqueeze(0)
        # code_encoder_hidden  [ 1， batchsize, hidden_size]
        token_encoder_output, token_encoder_hidden = self.code_encoder(method_code, code_encoder_hidden)
        summary_length = summary.size(1)
        decoder_outputs = torch.zeros(cf.batch_size, summary_length - 1, self.decoder.output_size)
        # decoder_outputs = torch.zeros(self.batch_size, summary_length, self.decoder.output_size)
        decoder_outputs = move_to_device(decoder_outputs)

        # Summary starts with <s>
        decoder_inputs = summary[:, 0]
        # decoder_hidden [batch_size, hidden_dim][1,10]
        # decoder_hidden = asts_encoder_hidden[0]
        # [2,1,10]--> [1,2,10]->[1,20]
        # decoder_hidden = asts_encoder_hidden.permute(1, 0, 2).reshape(self.batch_size, -1)
        decoder_hidden = token_encoder_hidden[0]  # [batch_size, hidden_size]
        for i in range(1, summary_length):
            # for i in range(1, summary_length+1): # no <s> and </s>
            decoder_output, decoder_hidden = self.decoder(decoder_inputs,
                                                          asts_encoder_hidden,
                                                          # asts_encoder_output,
                                                          token_encoder_hidden,
                                                          asts_encoder_output,
                                                          token_encoder_output,
                                                          decoder_hidden)

            # decoder_output = [batch size, vocab_size]
            decoder_outputs[:, i - 1, :] = decoder_output

            if i + 1 < summary_length:
                if use_teacher:
                    decoder_inputs = summary[:, i]
                else:
                    decoder_inputs = decoder_output.argmax(1).detach()

        return decoder_outputs
