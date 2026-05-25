"""
Residual structure diagnostic for APEC.

Answers three questions:
  Q1. Is there per-horizon bias? (mean residual per step h)
  Q2. Do consecutive H-step error windows correlate?
      (the key assumption behind multi-step correction)
  Q3. Does 1-step residual ACF decay quickly?
      (what the current APEC e_win captures)

Usage:
  python diagnose_residuals.py \
    --checkpoint_path ./checkpoints/<setting>/checkpoint.pth \
    --pred_len 96 --seq_len 96 --cycle 24 \
    --data ETTh1 --root_path ./dataset/ETT-small --data_path ETTh1.csv \
    --features M --enc_in 7

The script loads the checkpoint, runs inference on val+test, then prints
a concise summary and saves diagnose_residuals.png.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_provider.data_factory import data_provider
from models import GTR, GTRDLinear, GTRPatchTST, GTRiTransformer, CycleNet, \
    DLinear, NLinear, Linear, PatchTST, SegRNN, iTransformer, TimeXer, \
    Informer, Autoformer, Transformer

MODEL_DICT = {
    'GTR': GTR, 'GTRDLinear': GTRDLinear, 'GTRPatchTST': GTRPatchTST,
    'GTRiTransformer': GTRiTransformer, 'CycleNet': CycleNet,
    'DLinear': DLinear, 'NLinear': NLinear, 'Linear': Linear,
    'PatchTST': PatchTST, 'SegRNN': SegRNN, 'iTransformer': iTransformer,
    'TimeXer': TimeXer, 'Informer': Informer, 'Autoformer': Autoformer,
    'Transformer': Transformer,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def build_model(args, device):
    model = MODEL_DICT[args.model].Model(args).float().to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def forward(model, args, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, device):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    batch_cycle = batch_cycle.int().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(device)

    with torch.no_grad():
        if any(s in args.model for s in ('CycleNet', 'GTR')):
            out = model(batch_x, batch_cycle)
        elif any(s in args.model for s in ('Linear', 'MLP', 'SegRNN', 'TST')):
            out = model(batch_x)
        else:
            out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    f_dim = -1 if args.features == 'MS' else 0
    pred = out[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
    true = batch_y[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
    return pred, true  # [B, H, C]


def collect_errors(model, args, dataset, device):
    """Return errors array of shape [N, H, C]."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False,
                        num_workers=0, drop_last=False)
    all_errors = []
    for batch in loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = batch
        pred, true = forward(model, args, batch_x, batch_y,
                             batch_x_mark, batch_y_mark, batch_cycle, device)
        all_errors.append(true - pred)          # [B, H, C]
    return np.concatenate(all_errors, axis=0)   # [N, H, C]


def acf(x, max_lag=50):
    """ACF of a 1-D array."""
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    lags = range(1, min(max_lag + 1, n))
    return [np.dot(x[k:], x[:-k]) / (n * c0) for k in lags]


# ── diagnostics ───────────────────────────────────────────────────────────────

def q1_per_horizon_bias(errors):
    """errors: [N, H, C] → mean/std per horizon step, averaged over channels."""
    mean_h = errors.mean(axis=(0, 2))   # [H]
    std_h  = errors.std(axis=(0, 2))    # [H]
    return mean_h, std_h


def q2_consecutive_window_corr(errors, step=None):
    """
    Correlation between error window i and window i+step.
    errors: [N, H, C]
    step: how many samples to shift (default = pred_len samples, i.e., non-overlapping)
    Uses Pearson corr on the flattened H*C vector per sample.
    """
    N, H, C = errors.shape
    if step is None:
        step = max(1, H)            # non-overlapping windows

    flat = errors.reshape(N, H * C)  # [N, H*C]
    a = flat[:-step]
    b = flat[step:]

    # Pearson per-sample then average, or just global
    # Global: treat each (window_i, window_i+step) pair as one observation
    a_mean = a.mean(axis=1, keepdims=True)
    b_mean = b.mean(axis=1, keepdims=True)
    num = ((a - a_mean) * (b - b_mean)).mean(axis=1)
    den = (a - a_mean).std(axis=1) * (b - b_mean).std(axis=1) + 1e-8
    per_pair_corr = num / den
    return per_pair_corr   # [N-step]


def q3_one_step_acf(errors, max_lag=48):
    """
    ACF of 1-step residuals (errors[:, 0, :]).
    errors: [N, H, C]  → use step h=0, average ACF over channels.
    """
    one_step = errors[:, 0, :]   # [N, C]
    acfs = [acf(one_step[:, c], max_lag=max_lag) for c in range(one_step.shape[1])]
    return np.array(acfs).mean(axis=0)   # [max_lag]


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--model', default='GTR')
    p.add_argument('--data', default='ETTh1')
    p.add_argument('--root_path', default='./dataset/ETT-small')
    p.add_argument('--data_path', default='ETTh1.csv')
    p.add_argument('--features', default='M')
    p.add_argument('--target', default='OT')
    p.add_argument('--freq', default='h')
    p.add_argument('--embed', default='timeF')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=0)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--cycle', type=int, default=24)
    p.add_argument('--enc_in', type=int, default=7)
    p.add_argument('--dec_in', type=int, default=7)
    p.add_argument('--c_out', type=int, default=7)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--d_ff', type=int, default=2048)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--factor', type=int, default=1)
    p.add_argument('--moving_avg', type=int, default=25)
    p.add_argument('--activation', default='gelu')
    p.add_argument('--output_attention', action='store_true')
    p.add_argument('--model_type', default='mlp')
    p.add_argument('--use_revin', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--use_gpu', type=bool, default=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--use_multi_gpu', action='store_true')
    p.add_argument('--devices', default='0,1')
    p.add_argument('--splits', default='val,test',
                   help='comma-separated splits to analyse, e.g. val,test')
    p.add_argument('--out', default='diagnose_residuals.png')
    return p.parse_args()


def main():
    args = parse_args()
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu else 'cpu')
    print(f'Device: {device}')

    model = build_model(args, device)

    splits = [s.strip() for s in args.splits.split(',')]
    errors_by_split = {}
    for split in splits:
        dataset, _ = data_provider(args, split)
        errs = collect_errors(model, args, dataset, device)
        errors_by_split[split] = errs
        print(f'[{split}] errors shape: {errs.shape}  '
              f'mean={errs.mean():.5f}  std={errs.std():.5f}')

    # ── Q1: per-horizon bias ──────────────────────────────────────────────────
    print('\n── Q1: Per-horizon bias (mean residual per step) ──')
    for split, errs in errors_by_split.items():
        mean_h, std_h = q1_per_horizon_bias(errs)
        max_abs_bias = np.abs(mean_h).max()
        rms_bias = np.sqrt((mean_h ** 2).mean())
        rms_std  = np.sqrt((std_h  ** 2).mean())
        snr = rms_bias / (rms_std + 1e-8)
        print(f'  [{split}] max|bias|={max_abs_bias:.5f}  '
              f'rms_bias={rms_bias:.5f}  rms_std={rms_std:.5f}  '
              f'SNR(bias/std)={snr:.4f}')
        if snr < 0.05:
            print(f'    → residuals are essentially unbiased (SNR<0.05); '
                  f'mean-reversion correction will not help')
        else:
            print(f'    → detectable bias exists (SNR={snr:.3f}); '
                  f'correction may be useful')

    # ── Q2: consecutive window correlation ───────────────────────────────────
    print('\n── Q2: Consecutive H-step window correlation ──')
    for split, errs in errors_by_split.items():
        step = max(1, args.pred_len)
        corrs = q2_consecutive_window_corr(errs, step=step)
        mean_corr = corrs.mean()
        pct_pos   = (corrs > 0.1).mean() * 100
        print(f'  [{split}] mean_corr={mean_corr:.4f}  '
              f'frac(corr>0.1)={pct_pos:.1f}%  '
              f'p25/p50/p75={np.percentile(corrs,25):.3f}/'
              f'{np.percentile(corrs,50):.3f}/{np.percentile(corrs,75):.3f}')
        if mean_corr > 0.1:
            print(f'    → positive carry-over exists; '
                  f'multi-step correction from previous window is viable')
        elif mean_corr > 0.0:
            print(f'    → weak positive carry-over; correction is marginal')
        else:
            print(f'    → no carry-over; consecutive windows are independent; '
                  f'correction will not generalise')

    # ── Q3: 1-step ACF ───────────────────────────────────────────────────────
    print('\n── Q3: 1-step residual ACF (what current e_win captures) ──')
    for split, errs in errors_by_split.items():
        acf_vals = q3_one_step_acf(errs, max_lag=min(48, len(errs) - 1))
        sig_lags = int((np.abs(acf_vals) > 0.05).sum())
        print(f'  [{split}] ACF[1]={acf_vals[0]:.4f}  ACF[2]={acf_vals[1]:.4f}  '
              f'significant lags (|ACF|>0.05): {sig_lags} / {len(acf_vals)}')
        if sig_lags < 3:
            print(f'    → 1-step residuals are near white noise; '
                  f'e_win window carries almost no predictive signal')
        else:
            print(f'    → 1-step residuals have autocorrelation up to lag {sig_lags}; '
                  f'e_win window is informative')

    # ── plots ─────────────────────────────────────────────────────────────────
    n_splits = len(errors_by_split)
    fig, axes = plt.subplots(3, n_splits, figsize=(6 * n_splits, 10))
    if n_splits == 1:
        axes = axes[:, None]

    for col, (split, errs) in enumerate(errors_by_split.items()):
        H = errs.shape[1]

        # Q1: bias per horizon
        mean_h, std_h = q1_per_horizon_bias(errs)
        ax = axes[0, col]
        ax.fill_between(range(H), mean_h - std_h, mean_h + std_h,
                        alpha=0.25, label='±1 std')
        ax.plot(mean_h, label='mean bias', color='tab:blue')
        ax.axhline(0, color='k', linewidth=0.7, linestyle='--')
        ax.set_title(f'[{split}] Q1: per-horizon bias')
        ax.set_xlabel('horizon step h')
        ax.set_ylabel('residual')
        ax.legend(fontsize=8)

        # Q2: histogram of consecutive window correlations
        corrs = q2_consecutive_window_corr(errs, step=max(1, H))
        ax = axes[1, col]
        ax.hist(corrs, bins=40, edgecolor='k', linewidth=0.3)
        ax.axvline(corrs.mean(), color='tab:red', linestyle='--',
                   label=f'mean={corrs.mean():.3f}')
        ax.axvline(0, color='k', linewidth=0.7)
        ax.set_title(f'[{split}] Q2: consecutive window corr')
        ax.set_xlabel('Pearson r')
        ax.legend(fontsize=8)

        # Q3: ACF of 1-step residuals
        acf_vals = q3_one_step_acf(errs, max_lag=min(48, len(errs) - 1))
        ax = axes[2, col]
        ax.bar(range(1, len(acf_vals) + 1), acf_vals, width=0.8)
        conf = 1.96 / np.sqrt(len(errs))
        ax.axhline(conf,  color='tab:red', linestyle='--', linewidth=0.8)
        ax.axhline(-conf, color='tab:red', linestyle='--', linewidth=0.8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(f'[{split}] Q3: 1-step residual ACF')
        ax.set_xlabel('lag')
        ax.set_ylabel('ACF')

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'\nPlot saved → {args.out}')


if __name__ == '__main__':
    main()
