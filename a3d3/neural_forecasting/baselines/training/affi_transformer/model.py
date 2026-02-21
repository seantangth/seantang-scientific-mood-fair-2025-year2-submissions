"""
Affi Pre-LN Transformer Model Architecture
============================================
Pre-LayerNorm Transformer for neural signal forecasting on Monkey A (Affi).

Architecture overview:
    1. Per-channel processing with learned channel embeddings
    2. Learned positional embeddings for temporal position encoding
    3. Pre-LN Transformer encoder (norm_first=True) for temporal modeling
    4. Cross-channel multi-head attention for inter-channel interactions
    5. Linear prediction head for 10-step-ahead forecasting

Key design choices:
    - Pre-LN (norm_first=True): Places LayerNorm before attention/FFN rather than after.
      This stabilizes training and allows for higher learning rates compared to Post-LN.
    - GELU activation in the feedforward layers for smoother gradient flow.
    - Learned positional embeddings (vs. sinusoidal) since the sequence length is fixed at 10.
    - return_features=True mode exposes mean-pooled channel features for CORAL domain
      adaptation loss during training.

Reference: Adapted from standard Transformer encoder (Vaswani et al., 2017) with
Pre-LN modification (Xiong et al., 2020) for neural time-series forecasting.
"""

import torch
import torch.nn as nn


class TransformerForecasterPreLN(nn.Module):
    """Pre-LN Transformer forecaster for Affi (239 channels).

    Pipeline:
        1. Embed each channel with a learned embedding (h // 4 dims)
        2. Concatenate channel embedding with input features -> project to h
        3. Add learned positional embeddings
        4. Encode with Pre-LN Transformer encoder over time
        5. Take the last timestep representation for each channel
        6. Apply cross-channel multi-head self-attention
        7. Predict 10 future timesteps per channel via a linear head

    The model can optionally return mean-pooled channel features (via
    return_features=True) which are used during training for computing
    the CORAL domain adaptation loss.

    Args:
        n_ch: Number of electrode channels (239 for Affi).
        n_feat: Number of input features per channel per timestep.
        h: Hidden dimension (d_model) for the Transformer.
        n_layers: Number of Transformer encoder layers.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, n_ch: int, n_feat: int = 1, h: int = 384,
                 n_layers: int = 4, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()

        # Channel embedding: each channel gets a unique embedding
        self.channel_embed = nn.Embedding(n_ch, h // 4)

        # Project (n_feat + channel_embed_dim) -> h
        self.input_proj = nn.Linear(n_feat + h // 4, h)

        # Learned positional embeddings for T=10 input timesteps
        self.pos_embed = nn.Parameter(torch.randn(1, 10, h) * 0.02)

        # Pre-LN Transformer encoder (norm_first=True is the key difference)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=n_heads,
            dim_feedforward=h * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,  # Pre-LN: LayerNorm before attention/FFN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cross-channel attention
        self.cross_attn = nn.MultiheadAttention(h, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(h)

        # Prediction head: h -> 10 output steps
        self.pred_head = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 10),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: (B, T, C, F) - batch of input sequences.
            return_features: If True, also return mean-pooled channel features
                for domain adaptation loss (CORAL).

        Returns:
            predictions: (B, 10, C) - predicted neural activity.
            features (optional): (B, h) - mean-pooled channel features.
        """
        B, T, C, F = x.shape

        # Channel embeddings: (C, h//4) -> expand to (B, T, C, h//4)
        ch_emb = self.channel_embed(
            torch.arange(C, device=x.device)
        ).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Concatenate features with channel embeddings, then reshape for per-channel processing
        # (B, T, C, F + h//4) -> permute to (B, C, T, dim) -> reshape to (B*C, T, dim)
        x = torch.cat([x, ch_emb], -1).permute(0, 2, 1, 3).reshape(B * C, T, -1)

        # Project to hidden dim and add positional embeddings
        x = self.input_proj(x) + self.pos_embed[:, :T, :]

        # Transformer encoding over time
        x = self.transformer(x)  # (B*C, T, h)

        # Take last timestep, reshape to (B, C, h)
        x = x[:, -1, :].view(B, C, -1)

        # Cross-channel self-attention with residual connection
        x = self.attn_norm(x + self.cross_attn(x, x, x)[0])

        # Predict: (B, C, h) -> (B, C, 10) -> (B, 10, C)
        pred = self.pred_head(x).transpose(1, 2)

        if return_features:
            # Mean-pool over channels for domain adaptation loss
            features = x.mean(dim=1)  # (B, h)
            return pred, features
        return pred
