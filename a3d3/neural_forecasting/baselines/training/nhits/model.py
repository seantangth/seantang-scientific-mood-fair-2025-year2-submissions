"""
NHiTS (Neural Hierarchical Interpolation for Time Series) forecaster.

Architecture from v176/v192d: multi-scale hierarchical interpolation with
pooling at different resolutions. Each stack captures patterns at a different
temporal scale via MaxPool downsampling and frequency-domain upsampling.

Used for the Beignet dataset with 1 feature (feature 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NHiTSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps,
                 pool_kernel_size, n_freq_downsample, dropout=0.1):
        super().__init__()
        self.output_steps = output_steps
        self.pool_kernel_size = pool_kernel_size

        self.pooling = nn.MaxPool1d(
            kernel_size=pool_kernel_size, stride=pool_kernel_size, ceil_mode=True
        )
        self.pooled_size = (input_size + pool_kernel_size - 1) // pool_kernel_size

        self.mlp = nn.Sequential(
            nn.Linear(self.pooled_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )

        self.n_coeffs = max(1, output_steps // n_freq_downsample)
        self.backcast_proj = nn.Linear(hidden_size, input_size)
        self.forecast_proj = nn.Linear(hidden_size, self.n_coeffs)

    def forward(self, x, return_hidden=False):
        x_pooled = self.pooling(x.unsqueeze(1)).squeeze(1)

        if x_pooled.shape[1] < self.pooled_size:
            x_pooled = F.pad(x_pooled, (0, self.pooled_size - x_pooled.shape[1]))
        elif x_pooled.shape[1] > self.pooled_size:
            x_pooled = x_pooled[:, :self.pooled_size]

        h = self.mlp(x_pooled)
        backcast = self.backcast_proj(h)
        forecast_coeffs = self.forecast_proj(h)

        if self.n_coeffs < self.output_steps:
            forecast = F.interpolate(
                forecast_coeffs.unsqueeze(1),
                size=self.output_steps, mode='linear', align_corners=False,
            ).squeeze(1)
        else:
            forecast = forecast_coeffs[:, :self.output_steps]

        if return_hidden:
            return backcast, forecast, h
        return backcast, forecast


class NHiTSStack(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps, n_blocks=2,
                 pool_kernel_sizes=None, n_freq_downsamples=None, dropout=0.1):
        super().__init__()
        pool_kernel_sizes = pool_kernel_sizes or [1, 2][:n_blocks]
        n_freq_downsamples = n_freq_downsamples or [1, 2][:n_blocks]

        self.blocks = nn.ModuleList([
            NHiTSBlock(
                input_size, hidden_size, output_steps,
                pool_kernel_sizes[i % len(pool_kernel_sizes)],
                n_freq_downsamples[i % len(n_freq_downsamples)],
                dropout,
            )
            for i in range(n_blocks)
        ])

    def forward(self, x, return_hidden=False):
        residual = x
        total_forecast = 0
        hiddens = []

        for block in self.blocks:
            if return_hidden:
                backcast, forecast, h = block(residual, return_hidden=True)
                hiddens.append(h)
            else:
                backcast, forecast = block(residual)
            residual = residual - backcast
            total_forecast = total_forecast + forecast

        if return_hidden:
            return x - residual, total_forecast, hiddens
        return x - residual, total_forecast


class NHiTSForecaster(nn.Module):
    """
    Full NHiTS model with multiple stacks operating at different scales.

    Args:
        n_channels: Number of neural channels (e.g. 89 for Beignet).
        n_features: Number of input features per channel (1 for NHiTS).
        hidden_size: Hidden dimension for each block MLP.
        num_layers: Number of blocks per stack.
        n_stacks: Number of stacks (each with increasing pool sizes).
        seq_len: Input sequence length (number of time steps).
        dropout: Dropout rate.
        output_steps: Number of future time steps to predict.
    """

    def __init__(self, n_channels, n_features=1, hidden_size=128,
                 num_layers=2, n_stacks=2, seq_len=10, dropout=0.1,
                 output_steps=10):
        super().__init__()
        self.n_channels = n_channels
        self.output_steps = output_steps
        self.hidden_size = hidden_size

        self.stacks = nn.ModuleList([
            NHiTSStack(
                seq_len, hidden_size, output_steps, num_layers,
                pool_kernel_sizes=[pk * (i + 1) for pk in [1, 2]],
                n_freq_downsamples=[fd * (i + 1) for fd in [1, 2]],
                dropout=dropout,
            )
            for i in range(n_stacks)
        ])

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, T, C, F) input tensor.
            return_features: If True, also return aggregated hidden features
                             for domain adaptation (CORAL).

        Returns:
            pred: (B, output_steps, C) forecast.
            h_sample: (B, hidden_size) aggregated features (only if return_features=True).
        """
        B, T, C, F = x.shape
        x_flat = x[:, :, :, 0].transpose(1, 2).reshape(B * C, T)

        if return_features:
            all_hiddens = []
            total_forecast = 0
            for stack in self.stacks:
                _, forecast, hiddens = stack(x_flat, return_hidden=True)
                total_forecast = total_forecast + forecast
                all_hiddens.extend(hiddens)

            pred = total_forecast.view(B, C, self.output_steps).transpose(1, 2)

            # Aggregate hidden features: mean across all blocks, reshape to
            # (B, C, hidden), then mean across channels -> (B, hidden)
            h_cat = torch.stack(all_hiddens, dim=0).mean(dim=0)  # (B*C, hidden)
            h_sample = h_cat.view(B, C, -1).mean(dim=1)  # (B, hidden)
            return pred, h_sample
        else:
            total_forecast = sum(stack(x_flat)[1] for stack in self.stacks)
            return total_forecast.view(B, C, self.output_steps).transpose(1, 2)
