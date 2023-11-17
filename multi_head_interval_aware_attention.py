import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from utils import clones


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