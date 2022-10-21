import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from Model_Tools import clones


class FFN(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(features, exp_factor * features)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(exp_factor * features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class SelfAttn(nn.Module):
    def __init__(self, dropout):
        super(SelfAttn, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask):
        scale_term = math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, value)


class InrAttn(nn.Module):
    def __init__(self, dropout):
        super(InrAttn, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, str_mat, attn_mask):
        scale_term = math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term
        str_mat = str_mat.masked_fill(attn_mask == 0.0, -1e9)
        str_mat = F.softmax(str_mat, dim=-1)
        scores += str_mat
        if attn_mask is not None:
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, value)


class MHInrAttn(nn.Module):
    def __init__(self, features, n_head, dropout):
        super(MHInrAttn, self).__init__()
        self.d_h = features // n_head
        self.n_head = n_head
        self.linears = clones(nn.Linear(features, features), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, str_mat, attn_mask):
        b = x.size(0)
        query, key, value = [l(x).view(b, self.h, -1, self.d_h) for l, x in zip(self.linears, x)]
        scale_term = query.size(-1)
        str_mat = str_mat.masked_fill(attn_mask == 0.0, -1e9)
        str_mat = F.softmax(str_mat, dim=-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term + str_mat
        if attn_mask is not None:
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        x = torch.matmul(prob, value)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h*self.d_h)
        return self.linears[-1](x)

