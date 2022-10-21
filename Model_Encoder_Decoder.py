from Model_Attn import FFN, SelfAttn, InrAttn
from Model_Tools import SubLayerConnect, clones
import torch.nn as nn


class GeoEncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(GeoEncoderLayer, self).__init__()
        self.attn_layer = SelfAttn(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x):
        # (b ,n, l, d)
        x = self.sublayer[0](x, lambda x: self.attn_layer(x, x, x, None))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class GeoEncoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(GeoEncoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-2)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(EncoderLayer, self).__init__()
        self.inr_sa_layer = InrAttn(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x, str_mat, attn_mask):
        x = self.sublayer[0](x, lambda x: self.inr_sa_layer(x, x, x, str_mat, attn_mask))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class Encoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(Encoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, str_mat, attn_mask):
        for layer in self.layers:
            x = layer(x, str_mat, attn_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(DecoderLayer, self).__init__()
        self.inr_sa_layer = InrAttn(dropout)
        self.ffn_layer = FFN(features, exp_factor, dropout)
        self.sublayer = clones(SubLayerConnect(features), 2)

    def forward(self, x, mem, str_mat, mem_pad_mask):
        x = self.sublayer[0](x, lambda x: self.inr_sa_layer(x, mem, mem, str_mat, mem_pad_mask))
        x = self.sublayer[1](x, self.ffn_layer)
        return x


class Decoder(nn.Module):
    def __init__(self, features, layer, depth):
        super(Decoder, self).__init__()
        self.layers = clones(layer, depth)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, mem, str_mat, mem_pad_mask):
        for layer in self.layers:
            x = layer(x, mem, str_mat, mem_pad_mask)
        return self.norm(x)
