import torch
import torch.nn as nn


class TrendPlugin(nn.Module):
    """Global learnable correction analogous to GTR's Q.
    T[h, c] learns the systematic bias at horizon h for channel c."""

    def __init__(self, pred_len, n_channels):
        super().__init__()
        self.T = nn.Parameter(torch.zeros(pred_len, n_channels))

    def forward(self, y_hat):
        return y_hat + self.T.unsqueeze(0).expand(y_hat.shape[0], -1, -1)
