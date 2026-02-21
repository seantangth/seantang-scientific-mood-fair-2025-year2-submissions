"""
Training script for Beignet Public TCN (h=256) with CORAL domain adaptation.

Trains a TCN forecaster on the Beignet public neural recording data, using
CORAL loss and mean alignment loss to adapt source (train) features toward
the target (test) domain distribution.

Source notebook: colab_v204_beignet_pub_h256.ipynb

Usage:
    python train.py --data_dir /path/to/data --output_dir /path/to/output
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import TCNForecasterLarge


# ============================================================================
# CORAL Loss Functions
# ============================================================================

def coral_loss(source, target):
    """Correlation alignment loss between source and target feature distributions."""
    d = source.size(1)
    cs = (source - source.mean(0, keepdim=True)).T @ (source - source.mean(0, keepdim=True)) / (source.size(0) - 1 + 1e-8)
    ct = (target - target.mean(0, keepdim=True)).T @ (target - target.mean(0, keepdim=True)) / (target.size(0) - 1 + 1e-8)
    return ((cs - ct) ** 2).sum() / (4 * d)


def mean_alignment_loss(source, target):
    """Mean alignment loss between source and target feature distributions."""
    return ((source.mean(0) - target.mean(0)) ** 2).mean()


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_batch(x, y):
    """Apply augmentations to source batch during training.

    Args:
        x: (B, T, C, F) normalized input
        y: (B, T_out, C) normalized target

    Augmentations:
        1. Channel-wise mean shift (simulates baseline drift between domains)
        2. Amplitude scaling per channel (simulates gain drift)
        3. Gaussian noise (regularization)
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

def load_and_prepare_data(data_dir, test_dir, n_features=9, n_val=100):
    """Load Beignet data, normalize, and split into train/val.

    Args:
        data_dir: Path to directory containing train_data_beignet.npz.
        test_dir: Path to directory containing test_data_beignet_masked.npz.
        n_features: Number of input features per channel (9).
        n_val: Number of samples for validation (100).

    Returns:
        X_tr, Y_tr: Normalized training arrays.
        X_val, Y_val: Normalized validation arrays.
        X_target_n: Normalized target (test) input for CORAL.
        mean, std: Normalization statistics.
    """
    train_data = np.load(os.path.join(data_dir, 'train_data_beignet.npz'))['arr_0']
    test_public = np.load(os.path.join(test_dir, 'test_data_beignet_masked.npz'))['arr_0']

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


# ============================================================================
# Training Function
# ============================================================================

def train_coral_model(X_tr, Y_tr, X_val, Y_val, X_target_n, std,
                      h_size, n_layers, dropout, coral_w, mean_w, seed,
                      epochs=250, patience=30, batch_size=32, use_aug=True,
                      device=None):
    """Train a single TCN CORAL model with given config and seed.

    Args:
        X_tr, Y_tr: Normalized training data.
        X_val, Y_val: Normalized validation data.
        X_target_n: Normalized target domain input (for CORAL).
        std: Standard deviation array for denormalized MSE.
        h_size: Hidden dimension (256).
        n_layers: Number of TCN layers (3).
        dropout: Dropout rate (0.25).
        coral_w: Weight for CORAL loss (3.0).
        mean_w: Weight for mean alignment loss (1.0).
        seed: Random seed.
        epochs: Max training epochs.
        patience: Early stopping patience.
        batch_size: Batch size.
        use_aug: Whether to apply data augmentation.
        device: torch device.

    Returns:
        best_val: Best validation MSE (denormalized).
        best_state: Model state dict at best validation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    target_ds = TensorDataset(torch.FloatTensor(X_target_n))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    target_dl = DataLoader(target_ds, batch_size=batch_size, shuffle=True)

    n_ch = X_tr.shape[2]  # number of neural channels
    n_feat = X_tr.shape[3]  # number of features
    model = TCNForecasterLarge(n_ch, n_feat, h_size, n_layers, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, best_state, no_improve = float('inf'), None, 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        target_iter = iter(target_dl)
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            try:
                (xt,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_dl)
                (xt,) = next(target_iter)
            xt = xt.to(device)

            # Apply augmentation
            if use_aug:
                xb, yb = augment_batch(xb, yb)

            optimizer.zero_grad()
            pred, feat_src = model(xb, return_features=True)
            _, feat_tgt = model(xt, return_features=True)
            loss = (
                ((pred - yb) ** 2).mean()
                + coral_w * coral_loss(feat_src, feat_tgt)
                + mean_w * mean_alignment_loss(feat_src, feat_tgt)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += ((model(xb) - yb) ** 2).sum().item()
        val_mse = (val_loss / len(X_val)) * (std[..., 0] ** 2).mean()

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    elapsed = time.time() - t0
    print(f'  seed={seed} h={h_size} L={n_layers} do={dropout} '
          f'CORAL={coral_w} Mean={mean_w} '
          f'-> Val MSE: {best_val:.0f} ({n_params:,} params, ep {epoch+1}, {elapsed:.0f}s)')
    return best_val, best_state


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Beignet Public TCN h=256')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to train_data_neuro directory')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test_dev_input directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save model checkpoints')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 2024],
                        help='Random seeds for multi-seed training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Hyperparameters ---
    H_SIZE = 256
    N_LAYERS = 3
    DROPOUT = 0.25
    CORAL_W = 3.0
    MEAN_W = 1.0
    N_FEATURES = 9
    EPOCHS = 250
    PATIENCE = 30
    BATCH_SIZE = 32
    N_VAL = 100

    # --- Load Data ---
    X_tr, Y_tr, X_val, Y_val, X_target_n, mean, std = load_and_prepare_data(
        args.data_dir, args.test_dir, N_FEATURES, N_VAL
    )

    # --- Multi-Seed Training ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'\n=== Training {len(args.seeds)} seeds: h={H_SIZE} L={N_LAYERS} '
          f'do={DROPOUT} CORAL={CORAL_W} Mean={MEAN_W} ===')

    seed_results = []
    for s in args.seeds:
        val_mse, state = train_coral_model(
            X_tr, Y_tr, X_val, Y_val, X_target_n, std,
            H_SIZE, N_LAYERS, DROPOUT, CORAL_W, MEAN_W, s,
            epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
            device=device
        )
        seed_results.append((s, val_mse, state))

    # --- Save Checkpoints ---
    config = {
        'h_size': H_SIZE,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
        'coral_weight': CORAL_W,
        'mean_weight': MEAN_W,
        'n_features': N_FEATURES,
        'seeds': args.seeds,
    }

    for s, val_mse, state in seed_results:
        path = os.path.join(args.output_dir, f'model_tcn_seed{s}.pth')
        torch.save({
            'model_state_dict': state,
            'val_mse': val_mse,
            'config': config,
        }, path)
        fsize = os.path.getsize(path) / 1024
        print(f'Saved seed {s}: Val={val_mse:,.0f}, size={fsize:.0f}KB')

    # --- Summary ---
    print('\n=== Multi-seed Results ===')
    for s, val_mse, _ in seed_results:
        print(f'  seed={s}: Val MSE = {val_mse:,.0f}')
    vals = [r[1] for r in seed_results]
    print(f'Mean: {np.mean(vals):,.0f}, Std: {np.std(vals):,.0f}')
