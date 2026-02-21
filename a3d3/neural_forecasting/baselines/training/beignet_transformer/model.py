"""
Beignet Public Transformer Forecaster with Pre-LayerNorm (h=256).

Architecture: Transformer encoder with Pre-LN for stable training,
cross-channel attention, and prediction head.
- Channel embedding (h//4 dims) concatenated with input features
- Learnable positional embedding (max 10 timesteps)
- Pre-LN TransformerEncoder (norm_first=True) with GELU activation
- Cross-channel multi-head attention over the last time step
- Prediction head: Linear -> GELU -> Dropout -> Linear -> 10 output steps

Source notebook: colab_v213_stronger_transformer.ipynb
"""

import torch
import torch.nn as nn


class TransformerForecasterPreLN(nn.Module):
    """Transformer Forecaster with Pre-LayerNorm for Beignet Public domain.

    Pre-LN (norm_first=True) provides more stable training compared to
    the standard Post-LN Transformer.

    Args:
        n_ch: Number of neural channels (89 for Beignet).
        n_feat: Number of input features per channel per timestep (9).
        h: Hidden dimension size (256).
        n_layers: Number of transformer encoder layers (4).
        n_heads: Number of attention heads (4).
        dropout: Dropout rate (0.2).
    """
    def __init__(self, n_ch, n_feat=9, h=256, n_layers=4, n_heads=4, dropout=0.2):
        super().__init__()
        self.channel_embed = nn.Embedding(n_ch, h // 4)
        self.input_proj = nn.Linear(n_feat + h // 4, h)
        self.pos_embed = nn.Parameter(torch.randn(1, 10, h) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=h * 4,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True  # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cross_attn = nn.MultiheadAttention(h, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(h)
        self.pred_head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 10)
        )

    def forward(self, x, return_features=False):
        B, T, C, F = x.shape
        ch_emb = self.channel_embed(
            torch.arange(C, device=x.device)
        ).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        x = torch.cat([x, ch_emb], -1)
        x = x.permute(0, 2, 1, 3).reshape(B * C, T, -1)
        x = self.input_proj(x) + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        x = x[:, -1, :].view(B, C, -1)
        x = self.attn_norm(x + self.cross_attn(x, x, x)[0])
        pred = self.pred_head(x).transpose(1, 2)
        if return_features:
            return pred, x.mean(dim=1)
        return pred
