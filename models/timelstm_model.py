from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from torch import nn


class HistoricCurrent(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, model):
        super().__init__()
        self.model = model
        self.historic_model = TimeLSTM(embedding_dim, hidden_dim)
       
        self.fc_ct = nn.Linear(768, hidden_dim)
        self.fc_ct_attn = nn.Linear(768, hidden_dim//2)

        self.fc_concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_concat_attn = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(hidden_dim, 2)

    @staticmethod
    def combine_features(tweet_features, historic_features):
        return torch.cat((tweet_features, historic_features), 1)

    def forward(self, tweet_features, historic_features, lens, timestamp):
        if self.model == "tlstm":
            outputs = self.historic_model(historic_features, timestamp)
            tweet_features = F.relu(self.fc_ct(tweet_features))
            outputs = torch.mean(outputs, 1)
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = F.relu(self.fc_concat(combined_features))

        x = self.dropout(x)

        return self.final(x)


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        # assumes that batch_first is always true
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)

        h = h.cuda()
        c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs
