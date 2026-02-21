"""
Beignet Private TCN Forecaster (h=128).

Architecture: Causal TCN encoder with cross-channel attention.
Same architecture as the public TCN, but smaller (h=128) and without
CORAL domain adaptation (private data is small, no separate target domain).

- Channel embedding (h//4 dims) concatenated with input features
- TCN encoder with residual blocks, BatchNorm, GELU activation
- Cross-channel multi-head attention over the last time step
- Prediction head: Linear -> GELU -> Dropout -> Linear -> 10 output steps

Source notebook: colab_v202b_private_h128.ipynb
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.residual(x)
        x = self.dropout(self.activation(self.norm1(self.conv1(x))))
        x = self.dropout(self.activation(self.norm2(self.conv2(x))))
        return x + r


class TCNEncoder(nn.Module):
    def __init__(self, in_size, h_size, n_layers=4, k_size=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(in_size, h_size, 1)
        self.layers = nn.ModuleList([
            TCNBlock(h_size, h_size, k_size, 2**i, dropout)
            for i in range(n_layers)
        ])

    def forward(self, x):
        x = self.input_proj(x.transpose(1, 2))
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class TCNForecaster(nn.Module):
    """TCN Forecaster with h=128 for Beignet Private domains.

    Designed for small private datasets (82 and 76 samples respectively).
    No CORAL loss -- uses combined normalization across both private domains.

    Args:
        n_ch: Number of neural channels (89 for Beignet).
        n_feat: Number of input features per channel per timestep (1 or 9).
        h: Hidden dimension size (128).
        n_layers: Number of TCN layers (3).
        dropout: Dropout rate (0.3).
    """
    def __init__(self, n_ch, n_feat=1, h=128, n_layers=3, dropout=0.3):
        super().__init__()
        self.channel_embed = nn.Embedding(n_ch, h // 4)
        self.input_proj = nn.Linear(n_feat + h // 4, h)
        self.tcn = TCNEncoder(h, h, n_layers, 3, dropout)
        self.cross_attn = nn.MultiheadAttention(h, 4, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(h)
        self.pred_head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 10)
        )

    def forward(self, x):
        B, T, C, F = x.shape
        ch_emb = self.channel_embed(
            torch.arange(C, device=x.device)
        ).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        x = torch.cat([x, ch_emb], -1)
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, -1)
        x = self.tcn(self.input_proj(x))
        x = x[:, -1, :].view(B, C, -1)
        x = self.attn_norm(x + self.cross_attn(x, x, x)[0])
        return self.pred_head(x).transpose(1, 2)
