import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence


class TAPE(nn.Module):
    def __init__(self, dimension, dropout):
        super(TAPE, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.div_term = torch.exp(torch.arange(0, dimension, 2) * -(math.log(10000.0) / dimension)).to('cuda:0')

    def forward(self, x, position):
        tape = torch.zeros_like(x)
        tape[:, :, 0::2] = torch.sin(position[:].unsqueeze(-1) * self.div_term.unsqueeze(0))
        tape[:, :, 1::2] = torch.cos(position[:].unsqueeze(-1) * self.div_term.unsqueeze(0))
        x += Variable(tape, requires_grad=False)
        x = self.dropout(x)
        return x
