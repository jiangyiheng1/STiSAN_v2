import torch
import torch.nn as nn
import numpy as np


class KNNSamplerEvalNext(nn.Module):
    def __init__(self, loc_query_system, n_neighbor, user_visited_locs, exclude_visited):
        super(KNNSamplerEvalNext, self).__init__()
        self.loc_query_system = loc_query_system
        self.n_neighbor = n_neighbor
        self.exclude_visited = exclude_visited
        self.user_visited_locs = user_visited_locs

    def forward(self, trg_seq, n_neg, user):
        neg_samples = []
        trg_loc = trg_seq[0][1]
        ngb_locs = self.loc_query_system.get(trg_loc, self.n_neighbor)
        samples = []
        ngb_loc_idx = 0
        for _ in range(n_neg):
            sample = ngb_locs[ngb_loc_idx]
            while sample in self.user_visited_locs[user]:
                ngb_loc_idx += 1
                sample = ngb_locs[ngb_loc_idx]
            ngb_loc_idx += 1
            samples.append(sample)
        neg_samples.append(samples)
        neg_samples = torch.tensor(np.array(neg_samples))
        return neg_samples


class KNNSamplerEvalMulti(nn.Module):
    def __init__(self, loc_query_system, n_neighbor):
        super(KNNSamplerEvalMulti, self).__init__()
        self.loc_query_system = loc_query_system
        self.n_neighbor = n_neighbor

    def forward(self, trg_seq, n_neg):
        neg_samples = []
        pos_samples = [e[1] for e in trg_seq]
        for check_in in trg_seq:
            trg_loc = check_in[1]
            ngb_locs = self.loc_query_system.get(trg_loc, self.n_neighbor)
            samples = []
            ngb_loc_idx = 0
            for i in range(n_neg):
                sample = ngb_locs[ngb_loc_idx]
                while sample in pos_samples:
                    ngb_loc_idx += 1
                    sample = ngb_locs[ngb_loc_idx]
                ngb_loc_idx += 1
                samples.append(sample)
            neg_samples.append(samples)
        neg_samples = torch.tensor(np.array(neg_samples))
        return neg_samples
