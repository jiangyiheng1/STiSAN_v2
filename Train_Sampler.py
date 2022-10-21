import torch
import torch.nn as nn
import numpy as np


class KNNSamplerTrainNext(nn.Module):
    def __init__(self, loc_query_system, n_neighbor, user_visited_locs, exclude_visited):
        super(KNNSamplerTrainNext, self).__init__()
        self.loc_query_system = loc_query_system
        self.n_neighbor = n_neighbor
        self.exclude_visited = exclude_visited
        self.user_visited_locs = user_visited_locs

    def forward(self, trg_seq, n_neg, user):
        neg_samples = []
        for check_in in trg_seq:
            trg_loc = check_in[1]
            ngb_locs = self.loc_query_system.get(trg_loc, self.n_neighbor)
            samples = []
            if self.exclude_visited:
                for _ in range(n_neg):
                    sample = np.random.choice(ngb_locs)
                    while sample in self.user_visited_locs[user]:
                        sample = np.random.choice(ngb_locs)
                    samples.append(sample)
            else:
                samples = np.random.choice(ngb_locs, size=n_neg, replace=True)
            neg_samples.append(samples)
        neg_samples = torch.tensor(np.array(neg_samples))
        return neg_samples


class KNNSamplerTrainMulti(nn.Module):
    def __init__(self, loc_query_system, n_neighbor):
        super(KNNSamplerTrainMulti, self).__init__()
        self.loc_query_system = loc_query_system
        self.n_neighbor = n_neighbor

    def forward(self, trg_seq, n_neg):
        pos_samples = [e[1] for e in trg_seq]
        neg_samples = []
        for check_in in trg_seq:
            trg_loc = check_in[1]
            ngb_locs = self.loc_query_system.get(trg_loc, self.n_neighbor)
            samples = []
            for _ in range(n_neg):
                sample = np.random.choice(ngb_locs)
                while sample in pos_samples:
                    sample = np.random.choice(ngb_locs)
                samples.append(sample)
            neg_samples.append(samples)
        neg_samples = torch.tensor(np.array(neg_samples))
        return neg_samples
