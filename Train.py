import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import rearrange
from Train_Utils import reset_random_seed, LadderSampler
from Train_CollateFn import get_train_batch_next, get_train_batch_multi
from Evaluate import evaluate_next, evaluate_multi


def train_next(model, src_len, train_data, train_batch_size, n_epoch, loss_fn, optimizer, train_sampler, train_n_neg,
               eval_data, eval_batch_size, eval_sampler, eval_n_neg, processor, loc2gpscode, device):
    reset_random_seed(1)
    for epoch_idx in range(n_epoch):
        start_time = time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_data,
                                 batch_size=train_batch_size,
                                 sampler=LadderSampler(train_data, train_batch_size),
                                 num_workers=24,
                                 collate_fn=lambda e: get_train_batch_next(e,
                                                                           train_data,
                                                                           src_len,
                                                                           train_sampler,
                                                                           train_n_neg,
                                                                           processor,
                                                                           loc2gpscode))
        print("=====epoch {:>2d}=====".format(epoch_idx + 1))
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True, colour='blue')
        model.train()
        for batch_idx, (src_users, src_locs, src_gpscodes, src_times, src_lats, src_lons, src_data_size, trg_locs,
                        trg_gpscodes) in batch_iterator:
            src_user = src_users.to(device)
            src_loc = src_locs.to(device)
            src_gpscode = src_gpscodes.to(device)
            src_time = src_times.to(device)
            src_lat = src_lats.to(device)
            src_lon = src_lons.to(device)
            data_size = src_data_size.to(device)
            trg_loc = trg_locs.to(device)
            trg_gpscode = trg_gpscodes.to(device)
            optimizer.zero_grad()
            output = model(src_user, src_loc, src_gpscode, src_time, src_lat, src_lon, trg_loc, trg_gpscode, data_size)
            output = rearrange(rearrange(output, 'b (k n) -> b k n', k=1 + train_n_neg), 'b k n -> b n k')
            pos_scores, neg_scores = output.split([1, train_n_neg], -1)
            loss = loss_fn(pos_scores, neg_scores)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")

        epoch_time = time.time() - start_time
        avg_loss = running_loss / processed_batch
        print("time token: {:.2f} sec, avg. loss: {:.4f}".format(epoch_time, avg_loss))
    Hit_Ratio, NDCG = evaluate_next(model, src_len, eval_data, eval_batch_size, eval_sampler, eval_n_neg,
                                    processor, loc2gpscode, device)
    print("HR@5: {:.4f}".format(Hit_Ratio[4]), "NDCG@5: {:.4f}".format(NDCG[4]), "HR@10: {:.4f}".format(Hit_Ratio[9]),
          "NDCG@10: {:.4f}".format(NDCG[9]))


def train_multi(model, src_len, trg_len, train_data, train_batch_size, n_epoch, loss_fn, optimizer, train_sampler,
                train_n_neg, eval_data, eval_batch_size, eval_sampler, eval_n_neg, processor, loc2gpscode, device):
    reset_random_seed(1)
    for epoch_idx in range(n_epoch):
        start_time = time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_data,
                                 batch_size=train_batch_size,
                                 sampler=LadderSampler(train_data, train_batch_size),
                                 num_workers=24,
                                 collate_fn=lambda e: get_train_batch_multi(e,
                                                                            train_data,
                                                                            src_len,
                                                                            train_sampler,
                                                                            train_n_neg,
                                                                            processor,
                                                                            loc2gpscode))
        print("=====epoch {:>2d}=====".format(epoch_idx + 1))
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True, colour='blue')
        model.train()
        for batch_idx, (src_users, src_locs, src_gpscodes, src_times, src_lats, src_lons, src_data_size, trg_locs,
                        trg_gpscodes) in batch_iterator:
            src_user = src_users.to(device)
            src_loc = src_locs.to(device)
            src_gpscode = src_gpscodes.to(device)
            src_time = src_times.to(device)
            src_lat = src_lats.to(device)
            src_lon = src_lons.to(device)
            data_size = src_data_size.to(device)
            trg_loc = trg_locs.to(device)
            trg_gpscode = trg_gpscodes.to(device)
            optimizer.zero_grad()
            # [b, k*(1+n_neg)]
            output = model(src_user, src_loc, src_gpscode, src_time, src_lat, src_lon, trg_loc, trg_gpscode, data_size)
            output = rearrange(rearrange(output, 'b (n_neg k) -> b n_neg k', n_neg=train_n_neg + 1),
                               'b n_neg k -> b k n_neg')
            pos_scores, neg_scores = output.split([1, train_n_neg], -1)
            loss = loss_fn(pos_scores, neg_scores)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")

        epoch_time = time.time() - start_time
        avg_loss = running_loss / processed_batch
        print("time token: {:.2f} sec, avg. loss: {:.4f}".format(epoch_time, avg_loss))
    F1score = evaluate_multi(model, src_len, trg_len, eval_data, eval_batch_size, eval_sampler, eval_n_neg, processor,
                             loc2gpscode, device)
    print("F1score: {:.4f}".format(F1score))
