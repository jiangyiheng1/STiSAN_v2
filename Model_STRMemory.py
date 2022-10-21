import torch.nn as nn
from Model_Attn import FFN
from Model_Tools import SubLayerConnect, clones


class STRMemoryLayer(nn.Module):
    def __init__(self, src_len, trg_len, dropout):
        super(STRMemoryLayer, self).__init__()
        self.mem_v = FFN(trg_len, src_len, dropout)
        self.mem_h = FFN(src_len, trg_len, dropout)
        self.sublayer_1 = SubLayerConnect(trg_len)
        self.sublayer_2 = SubLayerConnect(src_len)

    def forward(self, x):
        # [b, n, k] -> [b, n, k*n] -> [b, n, k]
        x = self.sublayer_1(x, self.mem_v)
        # [b, k, n]
        x = x.transpose(-2, -1)
        # [b, k, n] -> [b, k, k*n] -> [b, k, n]
        x = self.sublayer_2(x, self.mem_h)
        # [b, n, k]
        x = x.transpose(-2, -1)
        return x


class STRMemory(nn.Module):
    def __init__(self, features, layer, depth):
        super(STRMemory, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        # [b, n, k]
        for layer in self.layers:
            x = layer(x)
        # [b, k, n]
        x = x.transpose(-2, -1)
        x = self.norm(x)
        return x
