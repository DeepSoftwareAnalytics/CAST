# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from util.LoggerUtil import debug_logger
from util.Config import Config as cf

class AttGRUDecoder(nn.Module):

    def __init__(self, hidden_size, vocab_size, emd_dim):
        super(AttGRUDecoder, self).__init__()
        self.output_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emd_dim)
        self.gru_cell = nn.GRUCell(input_size=emd_dim, hidden_size=hidden_size)
        self.predict = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_size, vocab_size))

    def forward(self, summary_token, encoder_hidden, encoder_output, prev_hidden_states):
        # summary_token is a batch of n-th nodes in the batch sequences
        # summary_token = [batch_size]
        # hidden = [1, batch_size, hidden_size]
        # encoder_outputs = [batch size, seq_len, hidden_size]

        # summary_embs = [batch_size, emb_dim]
        summary_embs = self.embedding(summary_token)

        # hidden_states = [batch_size, hidden_size]
        hidden_states = self.gru_cell(summary_embs, prev_hidden_states)

        debug_logger("AttGRUDecoder (1): summary_embs.shape %s, hidden_states.shape %s" % (
            str(summary_embs.shape), str(hidden_states.shape)))
        # hidden_states = [batch_size, 1, hidden_size]
        expand_hidden_states = hidden_states.unsqueeze(1)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        txt_attn = torch.bmm(expand_hidden_states, encoder_output.permute(0, 2, 1))
        # [batch_size, 1, seq_len]
        txt_attn = F.softmax(txt_attn, dim=2)
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        txt_context = torch.bmm(txt_attn, encoder_output)

        # [batch_size, 1, hidden_size * 2]
        context = torch.cat((txt_context, expand_hidden_states), dim=2)
        # [batch_size, hidden_size * 2]
        context = context.view(context.shape[0], -1)

        # [batch_size, vocab_size]
        output = self.predict(context)

        return output, hidden_states


# copy form CoCoGUM
class CodeAttnGRUDecoder(nn.Module):
    def __init__(self, vocab_size, emd_dim, hidden_size):
        super(CodeAttnGRUDecoder, self).__init__()
        self.output_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emd_dim)
        self.gru_cell = nn.GRUCell(input_size=emd_dim, hidden_size=hidden_size)
        self.predict = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_size, vocab_size))

    def forward(self, method_summary_token, sbt_encoder_hidden, code_encoder_hidden, sbt_encoder_output,
                code_encoder_output, prev_hidden_states):

        # method_summary_token is a batch of n-th nodes in the batch sequences
        # method_summary_token = [batch_size]
        # hidden = [1, batch_size, hidden_size]
        # encoder_outputs = [batch size, seq_len, hidden_size]

        summary_embs = self.embedding(method_summary_token)  # [batch_size, emb_dim]
        hidden_states = self.gru_cell(summary_embs, prev_hidden_states)  # [batch_size, hidden_size]

        debug_logger("ASTAttGRUDecoder (1): summary_embedding.shape %s, hidden_states.shape %s" % (str(summary_embs.shape), str(hidden_states.shape)))
        # hidden_states = [batch_size, 1, hidden_size]
        expand_hidden_states = hidden_states.unsqueeze(1)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        txt_attn = torch.bmm(expand_hidden_states, code_encoder_output.permute(0, 2, 1))
        txt_attn = F.softmax(txt_attn, dim=2)  # [batch_size, 1, seq_len]
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        txt_context = torch.bmm(txt_attn, code_encoder_output)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        ast_attn = torch.bmm(expand_hidden_states, sbt_encoder_output.permute(0, 2, 1))
        # [batch_size, 1, seq_len]
        ast_attn = F.softmax(ast_attn, dim=2)
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        ast_context = torch.bmm(ast_attn, sbt_encoder_output)

        # [batch_size, 1, hidden_size * 3]
        context = torch.cat((txt_context, expand_hidden_states, ast_context), dim=2)
        # [batch_size, hidden_size * 3]
        context = context.view(context.shape[0], -1)

        # [batch_size, vocab_size]
        output = self.predict(context)

        return output, hidden_states



# copy form CoCoGUM
class CodeAttnTransDecoder(nn.Module):
    def __init__(self, vocab_size, emd_dim, hidden_size):
        super(CodeAttnTransDecoder, self).__init__()
        self.output_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emd_dim)
        # self.gru_cell = nn.GRUCell(input_size=emd_dim, hidden_size=hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cf.feature_dim, nhead=cf.nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cf.nlayers)

        self.predict = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_size, vocab_size))

    # def forward(self, method_summary_token, sbt_encoder_hidden, code_encoder_hidden, sbt_encoder_output,
    #             code_encoder_output, prev_hidden_states):
    def forward(self, method_summary_token,  sbt_encoder_output,code_encoder_output, prev_hidden_states):

        # method_summary_token is a batch of n-th nodes in the batch sequences
        # method_summary_token = [batch_size]
        # hidden = [1, batch_size, hidden_size]
        # encoder_outputs = [batch size, seq_len, hidden_size]

        summary_embs = self.embedding(method_summary_token)  # [batch_size, emb_dim]
        # hidden_states = self.gru_cell(summary_embs, prev_hidden_states)  # [batch_size, hidden_size]
        batch_size = cf.batch_size
        summary_embs =summary_embs.view(1, batch_size, -1)
        hidden_states = self.transformer_decoder(summary_embs, prev_hidden_states)  # [batch_size, hidden_size]

        debug_logger("ASTAttGRUDecoder (1): summary_embedding.shape %s, hidden_states.shape %s" % (str(summary_embs.shape), str(hidden_states.shape)))
        # hidden_states = [ 1,batch_size, hidden_size]
        # expand_hidden_states = hidden_states.unsqueeze(1)
        expand_hidden_states = hidden_states.permute(1, 0, 2)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        txt_attn = torch.bmm(expand_hidden_states, code_encoder_output.permute(0, 2, 1))
        txt_attn = F.softmax(txt_attn, dim=2)  # [batch_size, 1, seq_len]
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        txt_context = torch.bmm(txt_attn, code_encoder_output)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        ast_attn = torch.bmm(expand_hidden_states, sbt_encoder_output.permute(0, 2, 1))
        # [batch_size, 1, seq_len]
        ast_attn = F.softmax(ast_attn, dim=2)
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        ast_context = torch.bmm(ast_attn, sbt_encoder_output)

        # [batch_size, 1, hidden_size * 3]
        context = torch.cat((txt_context, expand_hidden_states, ast_context), dim=2)
        # [batch_size, hidden_size * 3]
        context = context.view(context.shape[0], -1)

        # [batch_size, vocab_size]
        output = self.predict(context)

        return output, hidden_states