import argparse
import os
import torch
from exp.exp_main import Exp_Main
from exp.exp_apec import Exp_APEC
import random
import numpy as np

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2026, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='GTR',
                    help='model name, options: [Informer, Autoformer, ...]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# TQNet & CycleNet
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# APEC: Adaptive Post-hoc Error Correction
parser.add_argument('--use_apec', type=int, default=0, help='1: train/test APEC post-hoc error correction')
parser.add_argument('--apec_window', type=int, default=48, help='APEC residual context window')
parser.add_argument('--apec_hidden', type=int, default=128, help='APEC plug-in hidden size')
parser.add_argument('--apec_dropout', type=float, default=0.1, help='APEC plug-in dropout')
parser.add_argument('--apec_use_state_features', type=int, default=1, help='1: use trend/error state features in APEC plug-in')
parser.add_argument('--apec_epochs', type=int, default=15, help='APEC plug-in training epochs')
parser.add_argument('--apec_learning_rate', type=float, default=0.001, help='APEC plug-in learning rate')
parser.add_argument('--apec_alpha', type=float, default=0.1, help='APEC conformal miscoverage level')
parser.add_argument('--apec_val_plugin_ratio', type=float, default=0.65, help='fraction of official val used to train APEC plug-in')
parser.add_argument('--apec_val_gamma_ratio', type=float, default=0.15, help='fraction of official val used to select APEC gamma')
parser.add_argument('--apec_var_warmup', type=int, default=3, help='epochs to train delta before log variance')
parser.add_argument('--apec_mse_warmup', type=int, default=3, help='epochs to anneal auxiliary MSE after variance warmup')
parser.add_argument('--apec_mse_lambda', type=float, default=0.1, help='initial auxiliary MSE weight after variance warmup')
parser.add_argument('--apec_nll_weight', type=float, default=0.05, help='weight of Gaussian NLL after variance warmup')
parser.add_argument('--apec_delta_l2', type=float, default=0.0001, help='L2 penalty for APEC delta correction')
parser.add_argument('--apec_plugin_patience', type=int, default=5, help='early stopping patience for APEC plug-in')
parser.add_argument('--apec_gamma_step', type=float, default=0.1, help='grid step for validation-selected delta shrinkage')
parser.add_argument('--apec_gamma_mode', type=str, default='per_horizon',
                    help='APEC gamma mode, options: [scalar, per_horizon]')
parser.add_argument('--apec_gamma_min_improve', type=float, default=0.003,
                    help='per-horizon gamma: min relative MSE improvement to use correction (0.003 = 0.3%%)')
parser.add_argument('--apec_logvar_min', type=float, default=-7.0, help='minimum clamped log variance')
parser.add_argument('--apec_logvar_max', type=float, default=7.0, help='maximum clamped log variance')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_APEC if args.use_apec else Exp_Main


if args.is_training:
    for ii in range(args.itr):

        # setting record of experiments
        apec_tag = '_APEC_w{}_a{}'.format(args.apec_window, args.apec_alpha) if args.use_apec else ''
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.cycle,
            apec_tag,
            fix_seed)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict and not args.use_apec:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        elif args.do_predict:
            print('APEC does not implement do_predict yet; skipped unseen-future prediction.')

        torch.cuda.empty_cache()
else:
    ii = 0
    apec_tag = '_APEC_w{}_a{}'.format(args.apec_window, args.apec_alpha) if args.use_apec else ''
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        apec_tag,
        fix_seed)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
