import copy
import numpy as np
from math import floor
from nltk import ngrams
from collections import defaultdict
from torch.utils.data import Dataset
from torchtext.data import Field
from Data_Utils import latlon2quadkey, serialize
from Data_Location_Query_System import LocationQuerySystem


class LBSNData(Dataset):
    def __init__(self, path, min_loc_freq, min_user_freq, map_level):
        self.loc2idx = {'<pad>': 0}
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.idx2loc = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}
        self.loc2count = {}
        self.n_loc = 1
        self.build_location_vocab(path, min_loc_freq)
        self.n_user, self.user2idx, self.user_seq, self.n_gpscode, self.gpscode2idx, self.gpscode2loc = \
            self.process(path, min_user_freq, map_level)

    def build_location_vocab(self, path, min_loc_freq):
        print("Build Location Vocab...")
        for line in open(path, encoding='utf-8'):
            # 0    1        2    3   4
            # user location	time lon lat
            line = line.strip().split('\t')
            if line[0] == 'user_id:token':
                continue
            loc = line[1]
            # (lat, lon)
            gps = float(line[4]), float(line[3])
            self.add_location(loc, gps)
        if min_loc_freq > 0:
            self.n_loc = 1
            self.loc2idx = {'<pad>': 0}
            self.idx2loc = {0: '<pad>'}
            # (lat, lon)
            self.idx2gps = {0: (0.0, 0.0)}
            for loc in self.loc2count:
                if self.loc2count[loc] >= min_loc_freq:
                    self.add_location(loc, self.loc2gps[loc])
        self.loc2freq = np.zeros(self.n_loc - 1, dtype=np.int32)
        for idx, loc in self.idx2loc.items():
            if idx != 0:
                self.loc2freq[idx - 1] = self.loc2count[loc]

    def add_location(self, loc, gps):
        if loc not in self.loc2idx:
            self.loc2idx[loc] = self.n_loc
            self.loc2gps[loc] = gps
            self.idx2loc[self.n_loc] = loc
            self.idx2gps[self.n_loc] = gps
            if loc not in self.loc2count:
                self.loc2count[loc] = 1
            self.n_loc += 1
        else:
            self.loc2count[loc] += 1

    def process(self, path, min_user_freq, map_level):
        n_user = 1
        user2idx = {}
        user_seq = {}
        user_seq_array = list()
        n_gpscode = 1
        gpscode2idx = {}
        idx2gpscode = {}
        gpscode_idx2loc_idx = defaultdict(set)
        print("Encoding Quadkey Codes...")
        for line in open(path, encoding='utf-8'):
            # 0    1        2    3   4
            # user location	time lon lat
            line = line.strip().split('\t')
            if line[0] == 'user_id:token':
                continue
            user, loc, timestamp, lon, lat = line[0], line[1], float(line[2]), float(line[3]), float(line[4])
            if loc not in self.loc2idx:
                continue
            loc_idx = self.loc2idx[loc]
            gpscode = latlon2quadkey(lat, lon, map_level)

            if gpscode not in gpscode2idx:
                gpscode2idx[gpscode] = n_gpscode
                idx2gpscode[n_gpscode] = gpscode
                n_gpscode += 1
            gpscode_idx = gpscode2idx[gpscode]
            gpscode_idx2loc_idx[gpscode_idx].add(loc_idx)

            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([loc_idx, gpscode_idx, gpscode, timestamp, lat, lon])
        print("Filtering Cold Users...")
        for user, seq in user_seq.items():
            if len(seq) >= min_user_freq:
                user2idx[user] = n_user
                user_idx = n_user
                seq_new = list()
                tmp_set = set()
                cnt = 0
                for loc, _, gpscode, timestamp, lat, lon in sorted(seq, key=lambda e: e[3]):
                    if loc in tmp_set:
                        #               0         1    2        3          4    5    6
                        seq_new.append((user_idx, loc, gpscode, timestamp, lat, lon, True))
                    else:
                        seq_new.append((user_idx, loc, gpscode, timestamp, lat, lon, False))
                        tmp_set.add(loc)
                        cnt += 1
                if cnt > min_user_freq / 2:
                    n_user += 1
                    user_seq_array.append(seq_new)

        all_gpscodes = []
        print("Build Quadkey Vocab...")
        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                gpscode = check_in[2]
                gpscode_ngram = ' '.join([''.join(x) for x in ngrams(gpscode, 6)])
                gpscode_ngram = gpscode_ngram.split()
                all_gpscodes.append(gpscode_ngram)
                # u l g t lat lon bool
                user_seq_array[u][i] = (check_in[0], check_in[1], gpscode_ngram, check_in[3], check_in[4], check_in[5], check_in[6])

        self.loc2gpscode = ['NULL']
        for loc_idx in range(1, self.n_loc):
            lat, lon = self.idx2gps[loc_idx]
            gpscode = latlon2quadkey(lat, lon, map_level)
            gpscode_ngram = ' '.join([''.join(x) for x in ngrams(gpscode, 6)])
            gpscode_ngram = gpscode_ngram.split()
            self.loc2gpscode.append(gpscode_ngram)
            all_gpscodes.append(gpscode_ngram)

        self.GPSCODE = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            unk_token=None,
            preprocessing=str.split
        )
        self.GPSCODE.build_vocab(all_gpscodes)

        return n_user, user2idx, user_seq_array, n_gpscode, gpscode2idx, gpscode_idx2loc_idx

    def get_visited_locs(self):
        user_visited_locs = {}
        for u in range(len(self.user_seq)):
            seq = self.user_seq[u]
            user = seq[0][0]
            user_visited_locs[user] = set()
            for i in reversed(range(len(seq))):
                if not seq[i][-1]:
                    break
            user_visited_locs[user].add(seq[i][1])
            seq = seq[:i]
            for check_in in seq:
                user_visited_locs[user].add(check_in[1])
        return user_visited_locs

    def partition_next(self, src_len, trg_len):
        train_data = copy.copy(self)
        eval_data = copy.copy(self)
        train_seq = []
        eval_seq = []
        for user in range(len(self)):
            seq = self.user_seq[user]
            for i in reversed(range(len(seq))):
                if not seq[i][-1]:
                    break
            eval_trg_loc = seq[i: i + trg_len]
            eval_src_seq = seq[max(0, i - src_len): i]
            eval_seq.append((eval_src_seq, eval_trg_loc))

            for k in range(i):
                end_idx = i - k * trg_len
                if end_idx - src_len - trg_len > 0:
                    train_trg_seq = seq[end_idx - trg_len: end_idx]
                    train_src_seq = seq[end_idx - trg_len - src_len: end_idx - trg_len]
                    train_seq.append((train_src_seq, train_trg_seq))
                else:
                    train_trg_seq = seq[end_idx - trg_len: end_idx]
                    train_src_seq = seq[0: end_idx - trg_len]
                    train_seq.append((train_src_seq, train_trg_seq))
                    break
        train_data.user_seq = train_seq
        eval_data.user_seq = eval_seq
        return train_data, eval_data

    def partition_multi(self, src_len, trg_len):
        train_data = copy.copy(self)
        eval_data = copy.copy(self)
        train_seq = []
        eval_seq = []
        for user in range(len(self)):
            seq = self.user_seq[user]
            i = len(seq) - trg_len
            eval_trg_seq = seq[i: i + trg_len]
            eval_src_seq = seq[max(0, i - src_len): -trg_len]
            eval_seq.append((eval_src_seq, eval_trg_seq))

            for k in range(i):
                end_idx = i - k * trg_len
                if end_idx - src_len - trg_len > 0:
                    train_trg_seq = seq[end_idx - trg_len: end_idx]
                    train_src_seq = seq[end_idx - trg_len - src_len: end_idx - trg_len]
                    train_seq.append((train_src_seq, train_trg_seq))
                else:
                    train_trg_seq = seq[end_idx - trg_len: end_idx]
                    train_src_seq = seq[0: end_idx - trg_len]
                    train_seq.append((train_src_seq, train_trg_seq))
                    break
        train_data.user_seq = train_seq
        eval_data.user_seq = eval_seq
        return train_data, eval_data

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]