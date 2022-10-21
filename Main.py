from torch.optim import Adam
from Model_Predictor import Predictor
from Train import train_next, train_multi
from Train_Sampler import KNNSamplerTrainNext, KNNSamplerTrainMulti
from Evaluate_Sampler import KNNSamplerEvalNext, KNNSamplerEvalMulti
from Loss_Function import BCELoss
from Data_Utils import un_serialize
from Data_Location_Query_System import LocationQuerySystem
from Data_LBSN import LBSNData


if __name__ == "__main__":
    data_name = ''
    data_path = ''
    dataset = un_serialize(data_path)
    processor = dataset.GPSCODE
    loc2gpscode = dataset.loc2gpscode
    user_visited_locs = dataset.get_visited_locs()
    loc_query_path = ''
    loc_query_sys = LocationQuerySystem()
    loc_query_sys.load(loc_query_path)

    n_user = dataset.n_user
    n_loc = dataset.n_loc
    n_gpscode = dataset.n_loc
    k_t = ''
    k_d = ''
    dimension = ''
    exp_factor = ''
    depth = ''
    src_len = ''
    trg_len = ''
    dropout = ''
    device = ''
    model = Predictor(n_user, n_loc, n_gpscode, k_t, k_d, dimension, exp_factor, depth, src_len, trg_len, dropout, device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    train_batch_size = ''
    eval_batch_size = ''
    n_epoch = ''
    loss_fn = BCELoss()

    if trg_len == 1:
        train_data, eval_data = dataset.partition_next(src_len, trg_len)
        train_sampler = KNNSamplerTrainNext(loc_query_sys, 2000, user_visited_locs, True)
        train_n_neg = ''
        eval_sampler = KNNSamplerEvalNext(loc_query_sys, 2000, user_visited_locs, True)
        eval_n_neg = ''
        train_next(model, src_len, train_data, train_batch_size, n_epoch, loss_fn, optimizer, train_sampler,
                   train_n_neg, eval_data, eval_batch_size, eval_sampler, eval_n_neg, processor, loc2gpscode, device)
    else:
        train_data, eval_data = dataset.partition_multi(src_len, trg_len)
        train_sampler = KNNSamplerTrainMulti(loc_query_sys, 2000)
        train_n_neg = ''
        eval_sampler = KNNSamplerEvalMulti(loc_query_sys, 2000)
        eval_n_neg = ''
        train_multi(model, src_len, trg_len, train_data, train_batch_size, n_epoch, loss_fn, optimizer, train_sampler,
                    train_n_neg, eval_data, eval_batch_size, eval_sampler, eval_n_neg, processor, loc2gpscode, device)

