import argparse
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main


parser = argparse.ArgumentParser(description="CrossLink for Time Series Forecasting")

# random seed
parser.add_argument("--random_seed", type=int, default=2026, help="random seed")

# basic config
parser.add_argument("--is_training", type=int, required=True, default=1, help="status")
parser.add_argument("--model_id", type=str, required=True, default="test", help="model id")
parser.add_argument("--model", type=str, default="CrossLink",
                    help="model name, default: CrossLink")

# data loader
parser.add_argument("--data", type=str, required=True, default="ETTh1", help="dataset type")
parser.add_argument("--root_path", type=str, default="./data/ETT/", help="root path of the data file")
parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
parser.add_argument("--features", type=str, default="M",
                    help="forecasting task, options:[M, S, MS]")
parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
parser.add_argument("--freq", type=str, default="h", help="freq for time features encoding")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")

# forecasting task
parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
parser.add_argument("--label_len", type=int, default=0, help="start token length")
parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")

# CrossLink
parser.add_argument("--crosslink_lags", type=str, default="1,4,16,64",
                    help="comma-separated lag set, e.g. 1,4,16,64")
parser.add_argument("--crosslink_rank", type=int, default=16,
                    help="low-rank size; use 0 or negative for full matrices")

# Host MLP
parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
parser.add_argument("--embed", type=str, default="timeF",
                    help="time features encoding, options:[timeF, fixed, learned]")
parser.add_argument("--do_predict", action="store_true", help="whether to predict unseen future data")

# optimization
parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=30, help="train epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size of train input data")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--lradj", type=str, default="type3", help="adjust learning rate")
parser.add_argument("--pct_start", type=float, default=0.3, help="pct_start")
parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

# GPU
parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--use_multi_gpu", action="store_true", default=False, help="use multiple gpus")
parser.add_argument("--devices", type=str, default="0,1", help="device ids of multile gpus")
parser.add_argument("--test_flop", action="store_true", default=False, help="See utils/tools for usage")

args = parser.parse_args()

# Shared project plumbing still expects these attributes, but CrossLink scripts
# do not expose them as model hyperparameters.
args.cycle = 1
args.dropout = 0.0
args.patience = 5
args.use_revin = 0
args.output_attention = False

fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print("Args in experiment:")
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = "{}_{}_{}_ft{}_sl{}_pl{}_lags{}_rank{}_seed{}".format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.crosslink_lags.replace(",", "-"),
            args.crosslink_rank,
            fix_seed)

        exp = Exp(args)
        print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        exp.train(setting)

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting)

        if args.do_predict:
            print(">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = "{}_{}_{}_ft{}_sl{}_pl{}_lags{}_rank{}_seed{}".format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.crosslink_lags.replace(",", "-"),
        args.crosslink_rank,
        fix_seed)

    exp = Exp(args)
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
