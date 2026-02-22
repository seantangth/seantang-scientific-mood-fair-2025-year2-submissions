"""
NHiTS Utilities
=================
Shared helper functions for training and evaluating the NHiTS forecaster
with CORAL domain adaptation.

NHiTS uses only 1 feature (feature 0) from the Beignet dataset.

Includes:
    - CORAL and mean alignment domain adaptation losses
    - Data loading and normalization for Beignet (89 channels, 1 feature)
"""

import os

import numpy as np
import torch


# ============================================================================
# CORAL Domain Adaptation Loss
# ============================================================================

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute CORAL loss between source and target feature distributions.

    CORAL minimizes the difference of second-order statistics (covariance)
    between source and target domains.

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
    """Align first-order statistics (mean) between domains.

    Args:
        source: (N_s, d) source domain features.
        target: (N_t, d) target domain features.

    Returns:
        Scalar mean alignment loss (MSE between means).
    """
    return ((source.mean(0) - target.mean(0)) ** 2).mean()


# ============================================================================
# Dataset Preparation
# ============================================================================

def load_and_prepare_data(train_dir: str, test_dir: str, n_val: int = 100):
    """Load Beignet neural data with 1 feature for NHiTS, normalize, and split.

    NHiTS uses only feature 0 (primary neural signal) from the Beignet
    dataset. The data is z-score normalized per channel.

    Args:
        train_dir: Path to directory containing train_data_beignet.npz.
        test_dir: Path to directory containing test_data_beignet_masked.npz.
        n_val: Number of samples to hold out for validation.

    Returns:
        Dictionary with keys:
            - X_tr, X_val: Normalized training and validation inputs (1 feature).
            - Y_tr, Y_val: Normalized training and validation targets.
            - X_target: Normalized test (target domain) inputs.
            - mean, std: Normalization statistics (shape: 1, 1, n_channels, 1).
            - n_channels: Number of electrode channels (89 for Beignet).
    """
    train_data = np.load(
        os.path.join(train_dir, 'train_data_beignet.npz')
    )['arr_0']
    test_public = np.load(
        os.path.join(test_dir, 'test_data_beignet_masked.npz')
    )['arr_0']

    # NHiTS uses only feature 0
    X_train = train_data[:, :10, :, 0:1].astype(np.float32)
    Y_train = train_data[:, 10:, :, 0].astype(np.float32)
    X_target = test_public[:, :10, :, 0:1].astype(np.float32)

    # Z-score normalization per channel
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train_n = (X_train - mean) / std
    Y_train_n = (Y_train - mean[..., 0]) / std[..., 0]
    X_target_n = (X_target - mean) / std

    # Train/val split (last n_val samples for validation)
    X_tr, X_val = X_train_n[:-n_val], X_train_n[-n_val:]
    Y_tr, Y_val = Y_train_n[:-n_val], Y_train_n[-n_val:]

    n_channels = X_train.shape[2]
    print(f'Train: {len(X_tr)}, Val: {len(X_val)}, '
          f'Target: {len(X_target_n)}, Channels: {n_channels}')

    return {
        'X_tr': X_tr, 'X_val': X_val,
        'Y_tr': Y_tr, 'Y_val': Y_val,
        'X_target': X_target_n,
        'mean': mean, 'std': std,
        'n_channels': n_channels,
    }
