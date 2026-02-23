"""
Beignet Private Utilities
===========================
Shared helper functions for training and evaluating the Beignet Private
TCN model (h=256).

The private domains (2022-06-01 and 2022-06-02) are small datasets
(82 and 76 samples respectively). No CORAL domain adaptation is used
since there is no separate target domain. Combined normalization across
both private domains is applied for consistency.

Includes:
    - Data loading with combined normalization for both private domains
"""

import os

import numpy as np


# ============================================================================
# Dataset Preparation
# ============================================================================

def load_private_data(data_dir: str, n_features: int = 1):
    """Load both private domains and compute combined normalization.

    Loads the two private Beignet recording sessions and computes a
    single normalization (mean, std) across both, which was confirmed
    to work better than per-domain normalization.

    Args:
        data_dir: Path to directory containing the private .npz files:
            - train_data_beignet_2022-06-01_private.npz
            - train_data_beignet_2022-06-02_private.npz
        n_features: Number of input features per channel (1 or 9).

    Returns:
        train_priv1: Raw priv1 data array (82, 20, 89, 9).
        train_priv2: Raw priv2 data array (76, 20, 89, 9).
        mean_priv: Combined normalization mean (shape: 1, 1, 89, 9).
        std_priv: Combined normalization std (shape: 1, 1, 89, 9).
    """
    train_priv1 = np.load(
        os.path.join(data_dir, 'train_data_beignet_2022-06-01_private.npz')
    )['arr_0']
    train_priv2 = np.load(
        os.path.join(data_dir, 'train_data_beignet_2022-06-02_private.npz')
    )['arr_0']
    print(f'Priv1: {train_priv1.shape}')  # (82, 20, 89, 9)
    print(f'Priv2: {train_priv2.shape}')  # (76, 20, 89, 9)

    # Combined private normalization (confirmed better than per-domain)
    priv1_input = train_priv1[:, :10, :, :].astype(np.float32)
    priv2_input = train_priv2[:, :10, :, :].astype(np.float32)
    priv_all_input = np.concatenate([priv1_input, priv2_input], axis=0)
    mean_priv = priv_all_input.mean(axis=(0, 1), keepdims=True)
    std_priv = priv_all_input.std(axis=(0, 1), keepdims=True) + 1e-8

    print(f'Combined Private std (feat0) avg: {std_priv[..., 0].mean():.1f}')
    print(f'Total private samples: {len(priv_all_input)}')

    return train_priv1, train_priv2, mean_priv, std_priv
