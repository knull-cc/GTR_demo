import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoCycle(nn.Module):
    def __init__(self, cycle_len, channel_size, proto_num=8):
        super(ProtoCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.proto_num = proto_num

        self.prototypes = nn.Parameter(torch.randn(proto_num, cycle_len) * 0.02)
        self.mix_logits = nn.Parameter(torch.zeros(channel_size, proto_num))
        self.alpha = nn.Parameter(torch.zeros(1))

    def _period_table(self):
        mix = F.softmax(self.mix_logits, dim=-1)
        return mix @ self.prototypes

    def lookup(self, phase_index):
        table = self._period_table().t()
        phase_index = phase_index.long() % self.cycle_len
        return table[phase_index]

    def remove(self, x, end_index):
        phase_index = (
            end_index.view(-1, 1)
            - x.size(1)
            + torch.arange(x.size(1), device=x.device).view(1, -1)
        )
        cycle = self.lookup(phase_index)
        return x - self.alpha * cycle

    def future(self, end_index, pred_len):
        phase_index = end_index.view(-1, 1) + torch.arange(pred_len, device=end_index.device).view(1, -1)
        return self.alpha * self.lookup(phase_index)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.proto_num = getattr(configs, "proto_num", 8)

        self.proto_cycle = ProtoCycle(
            cycle_len=self.cycle_len,
            channel_size=self.enc_in,
            proto_num=self.proto_num
        )
        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )


    def forward(self, x, cycle_index):
        # RevIN normalize
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        residual = self.proto_cycle.remove(x, cycle_index)

        # Projection + MLP
        input_proj = self.input_proj(residual.permute(0, 2, 1))
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)
        output = output + self.proto_cycle.future(cycle_index, self.pred_len)

        # RevIN de-normalize
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
