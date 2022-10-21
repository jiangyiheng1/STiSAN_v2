import torch
from einops import rearrange
from Data_Utils import fix_seq_length, fix_gps_length


def get_eval_batch_next(batch, data_source, src_len, sampler, n_neg, processor, loc2gpscode):
    src_seq, trg_seq = zip(*batch)
    src_user, src_loc, src_time, src_gpscode, src_lat, src_lon, src_data_size = [], [], [], [], [], [], []
    for e in src_seq:
        # u, l, g, t, lat, lon, b
        u_, l_, g_, t_, lat_, lon_, _ = zip(*e)
        src_data_size.append(len(l_))
        u_ = fix_seq_length(u_, src_len)
        l_ = fix_seq_length(l_, src_len)
        t_ = fix_seq_length(t_, src_len)
        g_ = processor.numericalize(list(g_))
        g_ = fix_gps_length(g_, src_len)
        lat_ = fix_seq_length(lat_, src_len)
        lon_ = fix_seq_length(lon_, src_len)
        src_user.append(u_)
        src_loc.append(l_)
        src_time.append(t_)
        src_gpscode.append(g_)
        src_lat.append(lat_)
        src_lon.append(lon_)

    src_users = torch.stack(src_user)
    src_locs = torch.stack(src_loc)
    src_times = torch.stack(src_time)
    src_gpscodes = torch.stack(src_gpscode)
    src_lats = torch.stack(src_lat)
    src_lons = torch.stack(src_lon)
    src_data_size = torch.tensor(src_data_size)

    trg_loc, trg_gpscode = [], []
    for i, seq in enumerate(trg_seq):
        pos = torch.tensor([[e[1]] for e in seq])
        neg = sampler(seq, n_neg, user=seq[0][0])
        cat = torch.cat([pos, neg], dim=-1)
        trg_loc.append(cat)
        cat_gpscode = []
        for loc in range(cat.size(0)):
            tmp_gpscode = []
            for idx in cat[loc]:
                tmp_gpscode.append(loc2gpscode[idx])
            cat_gpscode.append(processor.numericalize(list(tmp_gpscode)))
        trg_gpscode.append(torch.stack(cat_gpscode))

    trg_locs = torch.stack(trg_loc)
    trg_gpscodes = torch.stack(trg_gpscode)
    trg_locs = rearrange(trg_locs, 'b k n_neg -> b n_neg k')
    trg_gpscodes = rearrange(trg_gpscodes, 'b k n_neg l -> b n_neg k l')
    return src_users, src_locs, src_gpscodes, src_times, src_lats, src_lons, src_data_size, trg_locs, trg_gpscodes


def get_eval_batch_multi(batch, data_source, src_len, sampler, n_neg, processor, loc2gpscode):
    src_seq, trg_seq = zip(*batch)
    src_user, src_loc, src_time, src_gpscode, src_lat, src_lon, src_data_size = [], [], [], [], [], [], []
    for e in src_seq:
        # u, l, g, t, lat, lon, b
        u_, l_, g_, t_, lat_, lon_, _ = zip(*e)
        src_data_size.append(len(l_))
        u_ = fix_seq_length(u_, src_len)
        l_ = fix_seq_length(l_, src_len)
        t_ = fix_seq_length(t_, src_len)
        g_ = processor.numericalize(list(g_))
        g_ = fix_gps_length(g_, src_len)
        lat_ = fix_seq_length(lat_, src_len)
        lon_ = fix_seq_length(lon_, src_len)
        src_user.append(u_)
        src_loc.append(l_)
        src_time.append(t_)
        src_gpscode.append(g_)
        src_lat.append(lat_)
        src_lon.append(lon_)

    src_users = torch.stack(src_user)
    src_locs = torch.stack(src_loc)
    src_times = torch.stack(src_time)
    src_gpscodes = torch.stack(src_gpscode)
    src_lats = torch.stack(src_lat)
    src_lons = torch.stack(src_lon)
    src_data_size = torch.tensor(src_data_size)

    trg_loc, trg_gpscode = [], []
    for i, seq in enumerate(trg_seq):
        pos = torch.tensor([[e[1]] for e in seq])
        neg = sampler(seq, n_neg)
        cat = torch.cat([pos, neg], dim=-1)
        trg_loc.append(cat)
        cat_gpscode = []
        for loc in range(cat.size(0)):
            tmp_gpscode = []
            for idx in cat[loc]:
                tmp_gpscode.append(loc2gpscode[idx])
            cat_gpscode.append(processor.numericalize(list(tmp_gpscode)))
        trg_gpscode.append(torch.stack(cat_gpscode))

    trg_locs = torch.stack(trg_loc)
    trg_gpscodes = torch.stack(trg_gpscode)
    trg_locs = rearrange(trg_locs, 'b k n_neg -> b n_neg k')
    trg_gpscodes = rearrange(trg_gpscodes, 'b k n_neg l -> b n_neg k l')
    return src_users, src_locs, src_gpscodes, src_times, src_lats, src_lons, src_data_size, trg_locs, trg_gpscodes
