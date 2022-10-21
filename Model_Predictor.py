import torch
from Model_Tools import Embedding
from Model_TAPE import TAPE
from Model_Encoder_Decoder import GeoEncoderLayer, GeoEncoder, EncoderLayer, Encoder, DecoderLayer, Decoder
from Model_STRMemory import STRMemoryLayer, STRMemory
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


class Predictor(nn.Module):
    def __init__(self, n_user, n_loc, n_gpscode, k_t, k_d, dimension, exp_factor, depth, src_len, trg_len, dropout, device):
        super(Predictor, self).__init__()
        self.emb_user = Embedding(n_user, dimension)
        self.emb_loc = Embedding(n_loc, dimension, True, True)
        self.emb_gpscode = Embedding(n_gpscode, dimension, True, True)
        
        self.geo_enc_layer = GeoEncoderLayer(dimension, exp_factor, dropout)
        self.geo_enc = GeoEncoder(dimension, self.geo_enc_layer, 2)
        
        self.tape = TAPE(dimension*2, dropout)

        self.k_t = k_t
        self.k_d = k_d
        self.enc_layer = EncoderLayer(dimension*2, exp_factor, dropout)
        self.enc = Encoder(dimension*2, self.enc_layer, depth)
        
        self.merge = nn.Linear(dimension, trg_len)
        self.str_mem_layer = STRMemoryLayer(src_len, trg_len, dropout)
        self.str_mem = STRMemory(src_len, self.str_mem_layer, depth)
        
        self.dec_layer = DecoderLayer(dimension*2, exp_factor, dropout)
        self.dec = Decoder(dimension*2, self.dec_layer, depth)
        
        self.src_len = src_len
        self.trg_len = trg_len
        self.device = device

    def get_time_aware_position(self, src_time, data_size):
        time_ = torch.clone(src_time)
        time_[:, 1:] = src_time[:, :-1]
        interval = (src_time - time_).masked_fill(src_time == 0, 0.0)
        sum_ = torch.sum(interval, dim=-1).unsqueeze(1)
        num_ = (data_size - 1).unsqueeze(1)
        avg_ = sum_ / num_
        interval /= avg_

        position = torch.zeros_like(src_time).to(self.device)
        position[:, 0] = 1.
        for k in range(1, src_time.size(1)):
            position[:, k] = position[:, k - 1] + interval[:, k] + 1
        position = position.masked_fill(src_time == 0, 0.0)
        return position
    
    def get_src_str_matrix(self, src_time, src_lat, src_lon, k_t, k_d):
        max_len = self.src_len
        time_seq = src_time
        lat_seq = src_lat
        lon_seq = src_lon
        time_mat_i = time_seq.unsqueeze(-1).expand([-1, max_len, max_len]).to(self.device)
        time_mat_j = time_seq.unsqueeze(1).expand([-1, max_len, max_len]).to(self.device)
        lat_mat_i = lat_seq.unsqueeze(-1).expand([-1, max_len, max_len]).deg2rad().to(self.device)
        lat_mat_j = lat_seq.unsqueeze(1).expand([-1, max_len, max_len]).deg2rad().to(self.device)
        lon_mat_i = lon_seq.unsqueeze(-1).expand([-1, max_len, max_len]).deg2rad().to(self.device)
        lon_mat_j = lon_seq.unsqueeze(1).expand([-1, max_len, max_len]).deg2rad().to(self.device)
        d_lat = lat_mat_i - lat_mat_j
        d_lon = lon_mat_i - lon_mat_j
        h = torch.sin(d_lat / 2) ** 2 + torch.cos(lat_mat_i) * torch.cos(lat_mat_j) * torch.sin(d_lon / 2) ** 2
        h = 2 * torch.asin(torch.sqrt(h))
        # day
        time_mat = torch.abs(time_mat_i - time_mat_j) / (3600. * 24)
        time_mat_max = (torch.ones_like(time_mat)*k_t)
        time_mat_ = torch.where(time_mat > time_mat_max, time_mat_max, time_mat) - time_mat
        # km
        spatial_mat = h * 6371
        spatial_mat_max = (torch.ones_like(spatial_mat) * k_d)
        spatial_mat_ = torch.where(spatial_mat > spatial_mat_max, spatial_mat_max, spatial_mat) - spatial_mat
        str_mat = time_mat_ + spatial_mat_
        return str_mat

    def get_src_pad_mask(self, data_size):
        mask = torch.zeros([data_size.size(0), self.src_len, self.src_len]).to(self.device)
        for i in range(data_size.size(0)):
            mask[i][0: data_size[i], 0: data_size[i]] = torch.ones(data_size[i], data_size[i])
        return mask

    def get_mem_mask(self, data_size):
        mask = torch.zeros([data_size.size(0), self.trg_len, self.src_len]).to(data_size.device)
        for i in range(data_size.size(0)):
            mask[i][:, 0: data_size[i]] = torch.ones([self.trg_len, data_size[i]])
        return mask

    def forward(self, src_user, src_loc, src_gpscode, src_time, src_lat, src_lon, trg_loc, trg_gpscode, data_size):
        # [b, n, d]
        src_loc_embedding = self.emb_loc(src_loc)
        # [b, n, l, d]
        src_gps_embedding = self.emb_gpscode(src_gpscode)
        # [b, n, d]
        src_gps_embedding = self.geo_enc(src_gps_embedding)
        # [b, n, 2*d]
        src = torch.cat([src_loc_embedding, src_gps_embedding], dim=-1)
        # [b, n]
        position = self.get_time_aware_position(src_time, data_size)
        # [b, n, 2d]
        src = self.tape(src, position)
        # [b, n, n]
        src_str_mat = self.get_src_str_matrix(src_time, src_lat, src_lon, self.k_t, self.k_d)
        # [b, n, n]
        src_pad_mask = self.get_src_pad_mask(data_size)
        # [b, n, 2d]
        enc_output = self.enc(src, src_str_mat, src_pad_mask)

        # [b, (1+n_neg), n, 2d]
        mem = enc_output.unsqueeze(1).repeat(1, trg_loc.size(1), 1, 1)
        # [b, n, d]
        src_user_embedding = self.emb_user(src_user)
        # [b, n, k]
        personalized_merge = self.merge(src_user_embedding)
        # [b, n, k]
        pred_str_mat = torch.matmul(src_str_mat, personalized_merge)
        # [b, k, n]
        pred_str_mat = self.str_mem(pred_str_mat)
        # [b, 1, k, n]
        pred_str_mat = pred_str_mat.unsqueeze(1)

        # [b, (1+n_neg), k, d]
        trg_loc_embedding = self.emb_loc(trg_loc)
        # [b, (1+n_neg), k, l, d]
        trg_gps_embedding = self.emb_gpscode(trg_gpscode)
        # [b, (1+n_neg), k, d]
        trg_gps_embedding = self.geo_enc(trg_gps_embedding)
        # [b, (1+n_neg), k, 2d]
        trg = torch.cat([trg_loc_embedding, trg_gps_embedding], dim=-1)
        # [b, 1, k, n]
        mem_pad_mask = self.get_mem_mask(data_size).unsqueeze(1)
        # [b, (1+n_neg), k, 2d]
        dec_output = self.dec(trg, mem, pred_str_mat, mem_pad_mask)

        # [b, (1+n_neg), k]
        output = torch.sum(dec_output * trg, dim=-1)
        # [b, k*(1+n_neg)]
        output = rearrange(output.contiguous(), 'b n_neg k -> b (n_neg k)')
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))