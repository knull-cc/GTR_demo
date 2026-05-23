import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_lags(lags):
    if isinstance(lags, str):
        return tuple(int(k.strip()) for k in lags.split(",") if k.strip())
    return tuple(int(k) for k in lags)


class CrossLink(nn.Module):
    """
    Plug-and-play lagged channel augmentation.

    Input:  x  [B, T, N]
    Output: x' [B, T, N * (K + 1)]
    """
    def __init__(self, N, lags=(1, 4, 16, 64), rank=None):
        super(CrossLink, self).__init__()
        self.N = N
        self.lags = _parse_lags(lags)
        self.rank = rank
        self.use_lowrank = rank is not None and 0 < rank < N

        if self.use_lowrank:
            self.U = nn.Parameter(torch.randn(N, rank) * 0.02)
            self.V = nn.Parameter(torch.randn(N, rank) * 0.02)
            self.r = nn.Parameter(torch.zeros(len(self.lags), rank))
        else:
            self.W = nn.Parameter(torch.zeros(len(self.lags), N, N))

    def _shift(self, x, k):
        if k <= 0:
            return x
        if k >= x.size(1):
            return torch.zeros_like(x)
        return F.pad(x[:, :-k, :], (0, 0, k, 0))

    def forward(self, x):
        outs = [x]
        for i, k in enumerate(self.lags):
            x_shifted = self._shift(x, k)
            if self.use_lowrank:
                z = x_shifted @ self.U
                z = z * self.r[i]
                e = z @ self.V.t()
            else:
                e = x_shifted @ self.W[i]
            outs.append(e)
        return torch.cat(outs, dim=-1)


class Model(nn.Module):
    """
    CrossLink + MLP host model.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = getattr(configs, "dropout", 0.0)
        self.use_revin = getattr(configs, "use_revin", 0)

        self.lags = _parse_lags(getattr(configs, "crosslink_lags", (1, 4, 16, 64)))
        rank = getattr(configs, "crosslink_rank", 16)
        rank = None if rank is None or rank <= 0 else rank

        self.crosslink = CrossLink(N=self.enc_in, lags=self.lags, rank=rank)
        self.channel_proj = nn.Linear(self.enc_in * (len(self.lags) + 1), self.enc_in)
        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

        self._init_channel_projection()

    def _init_channel_projection(self):
        with torch.no_grad():
            aug_weight = self.channel_proj.weight[:, self.enc_in:].clone()
            self.channel_proj.weight.zero_()
            self.channel_proj.bias.zero_()
            self.channel_proj.weight[:, :self.enc_in].copy_(torch.eye(self.enc_in))
            self.channel_proj.weight[:, self.enc_in:].copy_(aug_weight)

    def forward(self, x, *args, **kwargs):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_aug = self.crosslink(x)
        x = self.channel_proj(x_aug)

        x_input = x.permute(0, 2, 1)
        input_proj = self.input_proj(x_input)
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
