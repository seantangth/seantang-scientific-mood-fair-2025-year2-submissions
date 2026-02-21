"""
Affi TCN Model Architecture
============================
Temporal Convolutional Network for neural signal forecasting on Monkey A (Affi).

Architecture overview:
    1. Per-channel processing with learned channel embeddings
    2. Causal dilated convolutions (TCN) for temporal encoding
    3. Cross-channel multi-head attention for inter-channel interactions
    4. Linear prediction head for 10-step-ahead forecasting

Key design choices:
    - CausalConv1d ensures no information leakage from future timesteps
    - Exponentially increasing dilation (1, 2, 4, 8, ...) covers long-range dependencies
    - BatchNorm + GELU activation with proper residual connections (no activation after addition)
    - Channel embeddings allow the model to distinguish between 239 electrode channels

Reference: Based on the TCN architecture in Bai et al. (2018) "An Empirical Evaluation
of Generic Convolutional and Recurrent Networks for Sequence Modeling", adapted for
multi-channel neural forecasting with cross-channel attention.
"""

import torch
import torch.nn as nn
from typing import Optional


class CausalConv1d(nn.Module):
    """Causal convolution that ensures output depends only on past and current inputs.

    Pads the input on the left side so that the convolution output at time t
    only uses inputs from times <= t. The amount of left-padding is
    (kernel_size - 1) * dilation, and the excess right-side output is trimmed.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        dilation: Spacing between kernel elements (controls receptive field growth).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input tensor.
        Returns:
            (B, C_out, T) causally convolved output.
        """
        out = self.conv(x)
        # Remove the right-side padding to enforce causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Residual block for the TCN encoder.

    Structure:
        Input -> CausalConv1 -> BatchNorm -> GELU -> Dropout
              -> CausalConv2 -> BatchNorm -> GELU -> Dropout
              -> + Residual -> Output

    Note: The residual addition does NOT apply an activation afterward.
    An earlier version had `return self.activation(out + residual)` which
    caused triple activation and was identified as a bug. The fix is simply
    `return out + residual`.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor for this block.
        dropout: Dropout probability.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 convolution to match dimensions if in_channels != out_channels
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C_out, T)
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)

        # FIXED: no activation after residual addition (was a bug in earlier versions)
        return out + residual


class TCNEncoder(nn.Module):
    """Multi-layer TCN encoder with exponentially increasing dilation.

    Dilation pattern: 1, 2, 4, 8, ... (2^i for layer i)
    This gives the encoder an exponentially growing receptive field.

    Args:
        input_size: Dimensionality of the input features.
        hidden_size: Hidden dimension used throughout the TCN layers.
        num_layers: Number of stacked TCNBlock layers.
        kernel_size: Convolution kernel size for each block.
        dropout: Dropout probability.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 4,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)
        self.layers = nn.ModuleList([
            TCNBlock(hidden_size, hidden_size, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        # Receptive field = 1 + 2 * (kernel_size - 1) * sum(2^i for i in range(num_layers))
        self.receptive_field = 1 + (kernel_size - 1) * sum(2 ** i for i in range(num_layers)) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size)
        Returns:
            (B, T, hidden_size)
        """
        x = x.transpose(1, 2)       # (B, input_size, T)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)     # (B, T, hidden_size)


class TCNForecasterLarge(nn.Module):
    """TCN-based neural signal forecaster for Affi (239 channels).

    Pipeline:
        1. Embed each channel with a learned embedding (hidden_size // 4 dims)
        2. Concatenate channel embedding with input features -> project to hidden_size
        3. Run per-channel TCN encoding over time
        4. Take the last timestep representation for each channel
        5. Apply cross-channel multi-head self-attention
        6. Predict 10 future timesteps per channel via a linear head

    Args:
        n_channels: Number of electrode channels (239 for Affi, 89 for Beignet).
        n_features: Number of input features per channel per timestep.
        hidden_size: Hidden dimension for the TCN and attention layers.
        num_layers: Number of TCN blocks.
        kernel_size: Kernel size for causal convolutions.
        dropout: Dropout probability.
        output_steps: Number of future timesteps to predict.
    """

    def __init__(self, n_channels: int, n_features: int = 1, hidden_size: int = 384,
                 num_layers: int = 4, kernel_size: int = 3, dropout: float = 0.1,
                 output_steps: int = 10):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.output_steps = output_steps

        # Channel embedding
        self.channel_embed = nn.Embedding(n_channels, hidden_size // 4)

        # Project (n_features + channel_embed_dim) -> hidden_size
        self.input_proj = nn.Linear(n_features + hidden_size // 4, hidden_size)

        # TCN encoder (per channel)
        self.tcn = TCNEncoder(hidden_size, hidden_size, num_layers, kernel_size, dropout)

        # Cross-channel attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Prediction head: hidden_size -> output_steps
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_steps),
        )

    def forward(self, x: torch.Tensor, output_steps: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T_in, C, F) - batch of input sequences.
        Returns:
            predictions: (B, T_out, C) - predicted neural activity.
        """
        B, T, C, F = x.shape
        device = x.device

        # Channel embeddings: (C, hidden//4) -> (1, 1, C, hidden//4) -> (B, T, C, hidden//4)
        channel_ids = torch.arange(C, device=device)
        channel_emb = self.channel_embed(channel_ids).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Concatenate input features with channel embeddings
        # (B, T, C, F + hidden//4) -> (B*C, T, F + hidden//4)
        x_cat = torch.cat([x, channel_emb], dim=-1)
        x_flat = x_cat.permute(0, 2, 1, 3).reshape(B * C, T, -1)

        # Project and encode with TCN
        tcn_out = self.tcn(self.input_proj(x_flat))  # (B*C, T, hidden)

        # Take last timestep, reshape to (B, C, hidden)
        last = tcn_out[:, -1, :].view(B, C, -1)

        # Cross-channel self-attention
        attn_out, _ = self.cross_attn(last, last, last)
        feat = self.attn_norm(last + attn_out)

        # Predict: (B, C, hidden) -> (B, C, output_steps) -> (B, output_steps, C)
        predictions = self.pred_head(feat).transpose(1, 2)
        return predictions
