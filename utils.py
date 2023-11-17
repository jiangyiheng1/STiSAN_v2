import copy
import torch.nn as nn


def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class SubLayerConnect(nn.Module):
    def __init__(self, features):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        # (*, d)
        return x + sublayer(self.norm(x))