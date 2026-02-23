"""
Training script for Beignet Private TCN (h=256).

Trains separate TCN forecasters on each Beignet private domain
(2022-06-01 and 2022-06-02). Uses combined normalization across
both private domains for consistency.

No CORAL domain adaptation -- the private datasets are small
(82 and 76 samples) and there is no separate target domain to align to.

Source notebook: colab_v230_private_h256.ipynb

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

from model import TCNForecaster


# ============================================================================
# Dataset Preparation
# ============================================================================

def load_private_data(data_dir, n_features=1):
    """Load both private domains and compute combined normalization.

    Args:
        data_dir: Path to directory containing private .npz files.
        n_features: Number of input features per channel (1 or 9).

    Returns:
        train_priv1: Raw priv1 data array.
        train_priv2: Raw priv2 data array.
        mean_priv: Combined normalization mean.
        std_priv: Combined normalization std.
    """
    train_priv1 = np.load(os.path.join(data_dir, 'train_data_beignet_2022-06-01_private.npz'))['arr_0']
    train_priv2 = np.load(os.path.join(data_dir, 'train_data_beignet_2022-06-02_private.npz'))['arr_0']
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


# ============================================================================
# Training Function
# ============================================================================

def train_private_model(train_data, domain_mean, domain_std,
                        n_feat, h_size, n_layers, dropout,
                        seed=42, epochs=200, patience=25, batch_size=16, lr=5e-4,
                        device=None):
    """Train a single private TCN model with given config and seed.

    Args:
        train_data: Raw training data array (N, 20, 89, 9).
        domain_mean: Normalization mean (combined private).
        domain_std: Normalization std (combined private).
        n_feat: Number of input features (1 or 9).
        h_size: Hidden dimension (128).
        n_layers: Number of TCN layers (3).
        dropout: Dropout rate (0.3).
        seed: Random seed.
        epochs: Max training epochs.
        patience: Early stopping patience.
        batch_size: Batch size.
        lr: Learning rate.
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

    X_all = train_data[:, :10, :, :n_feat].astype(np.float32)
    Y_all = train_data[:, 10:, :, 0].astype(np.float32)

    dm = domain_mean[..., :n_feat]
    ds = domain_std[..., :n_feat]
    X_all_n = (X_all - dm) / ds
    Y_all_n = (Y_all - domain_mean[..., 0]) / domain_std[..., 0]

    # Random train/val split (1/6 for validation, min 8 samples)
    n = len(X_all_n)
    idx = np.random.permutation(n)
    n_val = max(8, n // 6)
    X_tr, X_val = X_all_n[idx[n_val:]], X_all_n[idx[:n_val]]
    Y_tr, Y_val = Y_all_n[idx[n_val:]], Y_all_n[idx[:n_val]]

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    n_ch = X_all.shape[2]  # 89
    model = TCNForecaster(n_ch, n_feat, h_size, n_layers, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, best_state, no_improve = float('inf'), None, 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
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
        val_mse = (val_loss / len(X_val)) * (domain_std[..., 0] ** 2).mean()

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    elapsed = time.time() - t0
    print(f'  seed={seed} h={h_size} L={n_layers} feat={n_feat} do={dropout}'
          f' -> Val MSE: {best_val:.0f} ({n_params:,} params, {n} samples, ep {epoch+1}, {elapsed:.0f}s)')
    return best_val, best_state


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Beignet Private TCN h=128')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to train_data_neuro directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save model checkpoints')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 2024],
                        help='Random seeds for multi-seed training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Hyperparameters ---
    N_FEAT = 1
    H_SIZE = 128
    N_LAYERS = 3
    DROPOUT = 0.3
    EPOCHS = 200
    PATIENCE = 25
    BATCH_SIZE = 16
    LR = 5e-4

    # --- Load Data ---
    train_priv1, train_priv2, mean_priv, std_priv = load_private_data(
        args.data_dir, N_FEAT
    )

    # --- Multi-Seed Training ---
    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        'n_feat': N_FEAT,
        'h_size': H_SIZE,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
        'seeds': args.seeds,
    }
    print(f'Config: {config}')

    # Train Priv1
    print(f'\n=== Priv1 ({len(train_priv1)} samples) ===')
    p1_results = []
    for s in args.seeds:
        val_mse, state = train_private_model(
            train_priv1, mean_priv, std_priv,
            N_FEAT, H_SIZE, N_LAYERS, DROPOUT,
            seed=s, epochs=EPOCHS, patience=PATIENCE,
            batch_size=BATCH_SIZE, lr=LR, device=device
        )
        p1_results.append((s, val_mse, state))

    # Train Priv2
    print(f'\n=== Priv2 ({len(train_priv2)} samples) ===')
    p2_results = []
    for s in args.seeds:
        val_mse, state = train_private_model(
            train_priv2, mean_priv, std_priv,
            N_FEAT, H_SIZE, N_LAYERS, DROPOUT,
            seed=s, epochs=EPOCHS, patience=PATIENCE,
            batch_size=BATCH_SIZE, lr=LR, device=device
        )
        p2_results.append((s, val_mse, state))

    # --- Save Checkpoints ---
    for s, val_mse, state in p1_results:
        path = os.path.join(args.output_dir, f'model_tcn_priv1_seed{s}.pth')
        torch.save({
            'model_state_dict': state,
            'val_mse': val_mse,
            'config': config,
        }, path)
        print(f'Priv1 seed {s}: Val={val_mse:.0f} -> {path}')

    for s, val_mse, state in p2_results:
        path = os.path.join(args.output_dir, f'model_tcn_priv2_seed{s}.pth')
        torch.save({
            'model_state_dict': state,
            'val_mse': val_mse,
            'config': config,
        }, path)
        print(f'Priv2 seed {s}: Val={val_mse:.0f} -> {path}')

    # Save normalization stats
    norm_path = os.path.join(args.output_dir, 'normalization_priv_combined.npz')
    np.savez(norm_path, mean=mean_priv, std=std_priv)
    print(f'\nNormalization saved to {norm_path}')

    # --- Summary ---
    print('\n=== Results ===')
    p1_vals = [r[1] for r in p1_results]
    p2_vals = [r[1] for r in p2_results]
    print(f'Priv1: mean={np.mean(p1_vals):.0f} std={np.std(p1_vals):.0f}')
    print(f'Priv2: mean={np.mean(p2_vals):.0f} std={np.std(p2_vals):.0f}')
    for i, s in enumerate(args.seeds):
        print(f'  seed={s}: P1={p1_vals[i]:.0f} P2={p2_vals[i]:.0f}')
