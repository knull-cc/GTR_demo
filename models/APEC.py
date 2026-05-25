import torch
import torch.nn as nn


class ChannelIndependentPlugIn(nn.Module):
    def __init__(self, window, pred_len, d_model=128, dropout=0.1, use_state_features=True):
        super(ChannelIndependentPlugIn, self).__init__()
        self.window = window
        self.pred_len = pred_len
        self.use_state_features = use_state_features
        self.state_dim = 11

        self.enc_x = nn.Linear(window, d_model)
        self.enc_e = nn.Linear(window, d_model)
        self.enc_y = nn.Linear(pred_len, d_model)
        if self.use_state_features:
            self.enc_state = nn.Sequential(
                nn.LayerNorm(self.state_dim),
                nn.Linear(self.state_dim, d_model),
                nn.GELU(),
            )
        fuse_in = 4 * d_model if self.use_state_features else 3 * d_model
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head_delta = nn.Linear(d_model, pred_len)
        self.head_logvar = nn.Linear(d_model, pred_len)
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)
        nn.init.zeros_(self.head_logvar.weight)
        nn.init.zeros_(self.head_logvar.bias)

    @staticmethod
    def _instance_norm(x):
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        return (x - mean) / std

    @staticmethod
    def _slope(x):
        steps = torch.linspace(-1.0, 1.0, x.size(-1), device=x.device, dtype=x.dtype)
        steps = steps.view(1, -1)
        centered = x - x.mean(dim=-1, keepdim=True)
        denom = torch.sum(steps ** 2).clamp_min(1e-6)
        return torch.sum(centered * steps, dim=-1, keepdim=True) / denom

    def _state_features(self, x_flat, e_flat, y_flat):
        x_slope = self._slope(x_flat)
        e_slope = self._slope(e_flat)
        y_slope = self._slope(y_flat)
        x_mean = x_flat.mean(dim=-1, keepdim=True)
        e_mean = e_flat.mean(dim=-1, keepdim=True)
        x_last = x_flat[:, -1:].contiguous()
        e_last = e_flat[:, -1:].contiguous()
        x_vol = x_flat.std(dim=-1, keepdim=True, unbiased=False)
        e_vol = e_flat.std(dim=-1, keepdim=True, unbiased=False)
        y_vol = y_flat.std(dim=-1, keepdim=True, unbiased=False)
        trend_gap = x_slope - y_slope
        return torch.cat(
            [x_mean, x_last, x_slope, x_vol,
             e_mean, e_last, e_slope, e_vol,
             y_slope, y_vol, trend_gap],
            dim=-1,
        )

    def encode(self, x_win, e_win, y_hat):
        batch_size, _, channels = x_win.shape
        x_flat = x_win.permute(0, 2, 1).reshape(batch_size * channels, self.window)
        e_flat = e_win.permute(0, 2, 1).reshape(batch_size * channels, self.window)
        y_flat = y_hat.permute(0, 2, 1).reshape(batch_size * channels, self.pred_len)
        state_features = self._state_features(x_flat, e_flat, y_flat)

        x_flat = self._instance_norm(x_flat)
        e_flat = self._instance_norm(e_flat)
        y_flat = self._instance_norm(y_flat)

        parts = [self.enc_x(x_flat), self.enc_e(e_flat), self.enc_y(y_flat)]
        if self.use_state_features:
            parts.append(self.enc_state(state_features))
        hidden = torch.cat(parts, dim=-1)
        return self.fuse(hidden), batch_size, channels

    def forward(self, x_win, e_win, y_hat):
        # mean-reversion baseline: broadcast mean recent residual across all horizons
        # computed on raw e_win before any normalization, so it carries actual scale
        e_base = e_win.permute(0, 2, 1).mean(dim=-1, keepdim=True)  # [B, C, 1]
        e_base = e_base.expand(-1, -1, self.pred_len)                # [B, C, H]

        hidden, batch_size, channels = self.encode(x_win, e_win, y_hat)
        delta_residual = self.head_delta(hidden).view(batch_size, channels, self.pred_len)
        logvar = self.head_logvar(hidden).view(batch_size, channels, self.pred_len)

        delta = (e_base + delta_residual).permute(0, 2, 1)  # [B, H, C]
        return delta, logvar.permute(0, 2, 1)
