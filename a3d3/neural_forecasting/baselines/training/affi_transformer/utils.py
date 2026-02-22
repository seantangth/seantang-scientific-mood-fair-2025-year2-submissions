"""
Affi Transformer Utilities
============================
Shared helper functions for training and evaluating the Affi Pre-LN
Transformer model.

Includes:
    - CORAL and mean alignment domain adaptation losses
    - Data augmentation for neural recording sequences
    - Data loading and normalization for Monkey A (Affi, 239 channels)
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
    source and target domains. This encourages the model to learn features
    that are invariant to the domain shift between training and test data.

    Reference: Sun & Saenko (2016) "Deep CORAL: Correlation Alignment for
    Deep Domain Adaptation"

    Args:
        source: (N_s, d) source domain features.
        target: (N_t, d) target domain features.

    Returns:
        Scalar CORAL loss.
    """
    d = source.size(1)
    # Source covariance
    src_centered = source - source.mean(0, keepdim=True)
    cs = (src_centered.T @ src_centered) / (source.size(0) - 1 + 1e-8)
    # Target covariance
    tgt_centered = target - target.mean(0, keepdim=True)
    ct = (tgt_centered.T @ tgt_centered) / (target.size(0) - 1 + 1e-8)
    # Frobenius norm of covariance difference, normalized by 4*d^2
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

    Three augmentation strategies are applied independently with probabilities:
        - Additive shift (p=0.5): Random per-channel DC offset (scale=0.15)
        - Multiplicative scaling (p=0.5): Random per-channel gain (scale=0.08)
        - Gaussian noise (p=0.3): Small additive noise (std=0.03)

    These augmentations simulate natural variability in neural recordings
    (e.g., electrode drift, gain changes, thermal noise).

    Args:
        x: (B, T, C, F) input sequences.
        y: (B, T_out, C) target sequences.

    Returns:
        Augmented (x, y) tuple.
    """
    # Additive shift: simulates baseline drift
    if torch.rand(1).item() < 0.5:
        shift = 0.15 * torch.randn(1, 1, x.shape[2], 1, device=x.device)
        x = x.clone()
        x[..., 0:1] = x[..., 0:1] + shift
        y = y + shift[..., 0].squeeze(0)

    # Multiplicative scaling: simulates gain variation
    if torch.rand(1).item() < 0.5:
        scale = 1.0 + 0.08 * torch.randn(1, 1, x.shape[2], 1, device=x.device)
        x = x * scale
        y = y * scale[..., 0].squeeze(0)

    # Gaussian noise: simulates measurement noise
    if torch.rand(1).item() < 0.3:
        x = x + 0.03 * torch.randn_like(x)

    return x, y


# ============================================================================
# Dataset Preparation
# ============================================================================

def load_and_prepare_data(data_dir: str, n_features: int = 3, n_val: int = 100):
    """Load Affi neural data and prepare train/val/target splits.

    The data is z-score normalized per channel using statistics computed
    from the training set input window (first 10 timesteps).

    Args:
        data_dir: Path to the directory containing train_data_neuro/ and
            test_dev_input/ subdirectories.
        n_features: Number of features to use (typically 3 for Transformer).
        n_val: Number of samples to hold out for validation.

    Returns:
        Dictionary with keys:
            - X_tr, X_val: Normalized training and validation inputs.
            - Y_tr, Y_val: Normalized training and validation targets.
            - X_target: Normalized test (target domain) inputs.
            - mean, std: Normalization statistics (shape: 1, 1, n_channels, n_features).
            - n_channels: Number of electrode channels (239 for Affi).
    """
    train_data = np.load(
        os.path.join(data_dir, 'train_data_neuro', 'train_data_affi.npz')
    )['arr_0']
    test_public = np.load(
        os.path.join(data_dir, 'test_dev_input', 'test_data_affi_masked.npz')
    )['arr_0']

    # Extract features and targets
    X_train = train_data[:, :10, :, :n_features].astype(np.float32)
    Y_train = train_data[:, 10:, :, 0].astype(np.float32)
    X_target = test_public[:, :10, :, :n_features].astype(np.float32)

    # Z-score normalization per channel
    # mean/std shape: (1, 1, n_channels, n_features)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train_n = (X_train - mean) / std
    Y_train_n = (Y_train - mean[..., 0]) / std[..., 0]
    X_target_n = (X_target - mean) / std

    # Train/val split (last n_val samples for validation)
    X_tr, X_val = X_train_n[:-n_val], X_train_n[-n_val:]
    Y_tr, Y_val = Y_train_n[:-n_val], Y_train_n[-n_val:]

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Target (test): {len(X_target_n)}")
    print(f"Channels: {X_train.shape[2]}, Features: {n_features}")

    return {
        'X_tr': X_tr, 'X_val': X_val,
        'Y_tr': Y_tr, 'Y_val': Y_val,
        'X_target': X_target_n,
        'mean': mean, 'std': std,
        'n_channels': X_train.shape[2],
    }
