import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        # [b, k, 1] -> [b, k]
        pos_scores = rearrange(pos_scores, 'b n l -> b (n l)')
        # log(sigmod(x)), [b ,k]
        pos_part = F.logsigmoid(pos_scores)
        # [b, k, n_neg] -> [b, k]
        neg_part = reduce(F.softplus(neg_scores), 'b n n_neg -> b n', 'mean')
        loss = -pos_part + neg_part
        return loss
