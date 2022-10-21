import numpy as np
import torch
from collections import Counter
from torch.utils.data import Sampler, DataLoader
from Train_Utils import reset_random_seed
from Evaluate_CollateFn import get_eval_batch_next, get_eval_batch_multi
from torchmetrics.classification import BinaryF1Score


def evaluate_next(model, src_len, eval_data, eval_batch_size, eval_sampler, eval_n_neg, processor, loc2gpscode, device):
    model.eval()
    reset_random_seed(1)
    loader = DataLoader(eval_data,
                        batch_size=eval_batch_size,
                        num_workers=24,
                        collate_fn=lambda e: get_eval_batch_next(e, eval_data, src_len, eval_sampler, eval_n_neg,
                                                                 processor, loc2gpscode))
    cnt = Counter()
    array = np.zeros(1 + eval_n_neg)
    with torch.no_grad():
        for _, (src_users, src_locs, src_gpscodes, src_times, src_lats, src_lons, src_data_size, trg_locs, trg_gpscodes) in enumerate(loader):
            src_user = src_users.to(device)
            src_loc = src_locs.to(device)
            src_gpscode = src_gpscodes.to(device)
            src_time = src_times.to(device)
            src_lat = src_lats.to(device)
            src_lon = src_lons.to(device)
            data_size = src_data_size.to(device)
            trg_loc = trg_locs.to(device)
            trg_gpscode = trg_gpscodes.to(device)
            scores = model(src_user, src_loc, src_gpscode, src_time, src_lat, src_lon, trg_loc, trg_gpscode, data_size)
            idx = scores.sort(descending=True, dim=1)[1]
            order = idx.topk(k=1, dim=1, largest=False)[1]
            cnt.update(order.squeeze().tolist())
    for k, v in cnt.items():
        array[k] = v
    hr = array.cumsum()
    ndcg = 1 / np.log2(np.arange(0, eval_n_neg + 1) + 2)
    ndcg = ndcg * array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()
    return hr, ndcg


def evaluate_multi(model, src_len, trg_len, eval_data, eval_batch_size, eval_sampler, eval_n_neg, processor, loc2gpscode, device):
    model.eval()
    reset_random_seed(1)
    loader = DataLoader(eval_data,
                        batch_size=eval_batch_size,
                        num_workers=24,
                        collate_fn=lambda e: get_eval_batch_multi(e, eval_data, src_len, eval_sampler, eval_n_neg,
                                                                  processor, loc2gpscode))
    F = BinaryF1Score()
    f1score = []
    with torch.no_grad():
        for _, (src_users, src_locs, src_gpscodes, src_times, src_lats, src_lons, src_data_size, trg_locs, trg_gpscodes) in enumerate(loader):
            src_user = src_users.to(device)
            src_loc = src_locs.to(device)
            src_gpscode = src_gpscodes.to(device)
            src_time = src_times.to(device)
            src_lat = src_lats.to(device)
            src_lon = src_lons.to(device)
            data_size = src_data_size.to(device)
            trg_loc = trg_locs.to(device)
            trg_gpscode = trg_gpscodes.to(device)
            scores = model(src_user, src_loc, src_gpscode, src_time, src_lat, src_lon, trg_loc, trg_gpscode, data_size)
            pred_idx = scores.topk(k=trg_len, dim=1, largest=True)[1]
            pred = torch.zeros_like(scores, dtype=torch.int)
            for i in range(pred.size(0)):
                pred[i, pred_idx[i]] = 1
            trg = torch.zeros_like(scores, dtype=torch.int)
            trg[:, :trg_len] = 1
            pred = pred.to('cpu')
            trg = trg.to('cpu')
            f1score.append(F(pred, trg).item())
    f1score = np.mean(f1score)
    return f1score
