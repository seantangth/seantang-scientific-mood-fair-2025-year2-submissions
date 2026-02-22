"""
Beignet TCN Utilities
======================
Shared helper functions for training and evaluating the Beignet Public
TCN model (h=256).

Includes:
    - CORAL and mean alignment domain adaptation losses
    - Data augmentation for neural recording sequences
    - Data loading and normalization for Beignet public domain (89 channels)
"""

import os

import numpy as np
import torch


# ============================================================================
# CORAL Domain Adaptation Loss
# ============================================================================

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute CORAL loss between source and target feature distributions.

    CORAL aligns the second-order statistics (covariance matrices) of the
    source and target domains.

    Reference: Sun & Saenko (2016) "Deep CORAL: Correlation Alignment for
    Deep Domain Adaptation"

    Args:
        source: (N_s, d) source domain features.
        target: (N_t, d) target domain features.

    Returns:
        Scalar CORAL loss.
    """
    d = source.size(1)
    cs = (source - source.mean(0, keepdim=True)).T @ \
         (source - source.mean(0, keepdim=True)) / (source.size(0) - 1 + 1e-8)
    ct = (target - target.mean(0, keepdim=True)).T @ \
         (target - target.mean(0, keepdim=True)) / (target.size(0) - 1 + 1e-8)
    return ((cs - ct) ** 2).sum() / (4 * d)


def mean_alignment_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean alignment loss between source and target features.

    Encourages the first-order statistics (means) of source and target
    feature distributions to match, complementing CORAL's second-order
    alignment.

    Args:
        source: (N_s, d) source domain features.
        target: (N_t, d) target domain features.

    Returns:
        Scalar mean alignment loss (MSE between means).
    """
    return ((source.mean(0) - target.mean(0)) ** 2).mean()


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_batch(x: torch.Tensor, y: torch.Tensor):
    """Apply stochastic data augmentation to a training batch.

    Augmentations applied independently:
        1. Channel-wise mean shift (p=0.5): Simulates baseline drift (scale=0.15)
        2. Amplitude scaling per channel (p=0.5): Simulates gain drift (scale=0.08)
        3. Gaussian noise (p=0.3): Regularization (std=0.03)

    Args:
        x: (B, T, C, F) normalized input sequences.
        y: (B, T_out, C) normalized target sequences.

    Returns:
        Augmented (x, y) tuple.
    """
    # 1. Channel-wise mean shift
    if torch.rand(1).item() < 0.5:
        shift = 0.15 * torch.randn(1, 1, x.shape[2], 1, device=x.device)
        x = x.clone()
        x[..., 0:1] = x[..., 0:1] + shift
        y = y + shift[..., 0].squeeze(0)

    # 2. Amplitude scaling per channel
    if torch.rand(1).item() < 0.5:
        scale = 1.0 + 0.08 * torch.randn(1, 1, x.shape[2], 1, device=x.device)
        x = x * scale
        y = y * scale[..., 0].squeeze(0)

    # 3. Gaussian noise
    if torch.rand(1).item() < 0.3:
        x = x + 0.03 * torch.randn_like(x)

    return x, y


# ============================================================================
# Dataset Preparation
# ============================================================================

def load_and_prepare_data(data_dir: str, test_dir: str, n_features: int = 9,
                          n_val: int = 100):
    """Load Beignet public neural data, normalize, and split into train/val.

    The data is z-score normalized per channel using statistics computed
    from the training set input window (first 10 timesteps).

    Args:
        data_dir: Path to directory containing train_data_beignet.npz.
        test_dir: Path to directory containing test_data_beignet_masked.npz.
        n_features: Number of input features per channel (9 for Beignet public).
        n_val: Number of samples to hold out for validation.

    Returns:
        X_tr, Y_tr: Normalized training arrays.
        X_val, Y_val: Normalized validation arrays.
        X_target_n: Normalized target (test) input for CORAL.
        mean: Normalization mean (shape: 1, 1, n_channels, n_features).
        std: Normalization std (shape: 1, 1, n_channels, n_features).
    """
    train_data = np.load(
        os.path.join(data_dir, 'train_data_beignet.npz')
    )['arr_0']
    test_public = np.load(
        os.path.join(test_dir, 'test_data_beignet_masked.npz')
    )['arr_0']

    X_train = train_data[:, :10, :, :n_features].astype(np.float32)
    Y_train = train_data[:, 10:, :, 0].astype(np.float32)
    X_target = test_public[:, :10, :, :n_features].astype(np.float32)

    # Normalization: mean/std computed from X_train over axes (0,1)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train_n = (X_train - mean) / std
    Y_train_n = (Y_train - mean[..., 0]) / std[..., 0]
    X_target_n = (X_target - mean) / std

    # Validation split: last n_val samples
    X_tr, X_val = X_train_n[:-n_val], X_train_n[-n_val:]
    Y_tr, Y_val = Y_train_n[:-n_val], Y_train_n[-n_val:]

    print(f'Train: {len(X_tr)}, Val: {len(X_val)}, Target: {len(X_target_n)}')
    print(f'X shape: {X_tr.shape}, Y shape: {Y_tr.shape}')

    return X_tr, Y_tr, X_val, Y_val, X_target_n, mean, std
