import torch
import copy
import torch.nn as nn
import torch.nn.functional as F


def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=False):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(inputs, self.lookup_table, self.padding_idx, None, 2, False, False)
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class SubLayerConnect(nn.Module):
    def __init__(self, features):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        # (*, d)
        return x + sublayer(self.norm(x))
