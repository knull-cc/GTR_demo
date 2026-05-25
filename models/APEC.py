import torch
import torch.nn as nn


class ChannelIndependentPlugIn(nn.Module):
    def __init__(self, window, pred_len, d_model=128, dropout=0.1):
        super(ChannelIndependentPlugIn, self).__init__()
        self.window = window
        self.pred_len = pred_len

        self.enc_x = nn.Linear(window, d_model)
        self.enc_e = nn.Linear(window, d_model)
        self.enc_y = nn.Linear(pred_len, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head_delta = nn.Linear(d_model, pred_len)
        self.head_logvar = nn.Linear(d_model, pred_len)

    @staticmethod
    def _instance_norm(x):
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        return (x - mean) / std

    def encode(self, x_win, e_win, y_hat):
        batch_size, _, channels = x_win.shape
        x_flat = x_win.permute(0, 2, 1).reshape(batch_size * channels, self.window)
        e_flat = e_win.permute(0, 2, 1).reshape(batch_size * channels, self.window)
        y_flat = y_hat.permute(0, 2, 1).reshape(batch_size * channels, self.pred_len)

        x_flat = self._instance_norm(x_flat)
        e_flat = self._instance_norm(e_flat)
        y_flat = self._instance_norm(y_flat)

        hidden = torch.cat(
            [self.enc_x(x_flat), self.enc_e(e_flat), self.enc_y(y_flat)],
            dim=-1,
        )
        return self.fuse(hidden), batch_size, channels

    def forward(self, x_win, e_win, y_hat):
        hidden, batch_size, channels = self.encode(x_win, e_win, y_hat)
        delta = self.head_delta(hidden).view(batch_size, channels, self.pred_len)
        logvar = self.head_logvar(hidden).view(batch_size, channels, self.pred_len)
        return delta.permute(0, 2, 1), logvar.permute(0, 2, 1)
