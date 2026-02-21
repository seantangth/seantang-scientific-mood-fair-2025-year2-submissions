"""
Affi Pre-LN Transformer Training Script
=========================================
Train a Pre-LN Transformer on Monkey A (Affi) neural data with optional
CORAL domain adaptation loss.

The CORAL (CORrelation ALignment) loss aligns the second-order statistics of
source (training) and target (test) domain feature distributions. Combined
with mean alignment loss, this encourages the model to learn features that
generalize across recording sessions.

This script trains both CORAL-enabled and CORAL-disabled ("orig") versions.
In the final ensemble, both variants are blended to improve robustness.

Usage:
    # Train with CORAL (default)
    python train.py --data_dir /path/to/data --coral_weight 3.0

    # Train without CORAL
    python train.py --data_dir /path/to/data --coral_weight 0.0

    # Train with 5 seeds for ensembling
    python train.py --data_dir /path/to/data --seeds 42,123,456,789,2024
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from model import TransformerForecasterPreLN


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
        Dictionary with normalized arrays and normalization parameters.
    """
    train_data = np.load(os.path.join(data_dir, 'train_data_neuro', 'train_data_affi.npz'))['arr_0']
    test_public = np.load(os.path.join(data_dir, 'test_dev_input', 'test_data_affi_masked.npz'))['arr_0']

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


# ============================================================================
# Training Function
# ============================================================================

def train_one_seed(
    data: dict,
    n_features: int = 3,
    h: int = 384,
    n_layers: int = 4,
    n_heads: int = 8,
    dropout: float = 0.2,
    coral_weight: float = 3.0,
    mean_weight: float = 1.0,
    seed: int = 42,
    epochs: int = 200,
    patience: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    warmup_epochs: int = 10,
    device: torch.device = None,
):
    """Train a single Transformer model with a given random seed.

    The training loop includes:
        - Mixed precision training (AMP) for memory efficiency
        - Linear warmup + cosine decay learning rate schedule
        - Data augmentation (shift, scale, noise)
        - CORAL + mean alignment domain adaptation loss (optional)
        - Gradient clipping (max norm = 1.0)
        - Early stopping based on validation MSE

    Args:
        data: Dictionary from load_and_prepare_data().
        n_features: Number of input features (typically 3).
        h: Transformer hidden dimension (d_model).
        n_layers: Number of Transformer encoder layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        coral_weight: Weight for CORAL loss (0 to disable domain adaptation).
        mean_weight: Weight for mean alignment loss.
        seed: Random seed for reproducibility.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience in epochs.
        batch_size: Training batch size. 16 fits well for h=384 on A100.
        lr: Peak learning rate (after warmup).
            5e-4 was used for Transformer (higher than TCN's 1e-4 due to
            Pre-LN stabilization).
        warmup_epochs: Number of linear warmup epochs.
        device: Torch device.

    Returns:
        Tuple of (best_val_mse, best_model_state_dict).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create data loaders
    train_ds = TensorDataset(torch.FloatTensor(data['X_tr']), torch.FloatTensor(data['Y_tr']))
    val_ds = TensorDataset(torch.FloatTensor(data['X_val']), torch.FloatTensor(data['Y_val']))
    target_ds = TensorDataset(torch.FloatTensor(data['X_target']))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    target_dl = DataLoader(target_ds, batch_size=batch_size, shuffle=True)

    # Initialize model
    n_channels = data['n_channels']
    model = TransformerForecasterPreLN(
        n_ch=n_channels, n_feat=n_features,
        h=h, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: TransformerForecasterPreLN(ch={n_channels}, feat={n_features}, "
          f"h={h}, L={n_layers}, heads={n_heads}) -> {n_params:,} params ({n_params/1e6:.2f}M)")

    # Optimizer: AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Mixed precision training
    scaler = GradScaler()

    # Learning rate schedule: linear warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val = float('inf')
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        target_iter = iter(target_dl)
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            # Get a batch of target domain data for CORAL
            try:
                (xt,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_dl)
                (xt,) = next(target_iter)
            xt = xt.to(device)

            # Data augmentation
            xb, yb = augment_batch(xb, yb)

            optimizer.zero_grad()

            with autocast():
                # Forward pass with feature extraction for CORAL
                pred, feat_src = model(xb, return_features=True)
                _, feat_tgt = model(xt, return_features=True)

                # MSE prediction loss
                mse_loss = ((pred - yb) ** 2).mean()

                # CORAL + mean alignment domain adaptation loss
                if coral_weight > 0:
                    c_loss = coral_loss(feat_src.float(), feat_tgt.float())
                    m_loss = mean_alignment_loss(feat_src.float(), feat_tgt.float())
                    loss = mse_loss + coral_weight * c_loss + mean_weight * m_loss
                else:
                    loss = mse_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                with autocast():
                    val_loss += ((model(xb) - yb) ** 2).sum().item()

        # Convert normalized MSE back to original scale for interpretability
        std = data['std']
        val_mse = (val_loss / len(data['X_val'])) * (std[..., 0] ** 2).mean()

        # Early stopping
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            tag = 'CORAL' if coral_weight > 0 else 'orig'
            print(f"    [{tag}] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_loss/n_batches:.4f} | "
                  f"Val MSE: {val_mse:,.0f} | "
                  f"Best: {best_val:,.0f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"{elapsed:.0f}s")

    elapsed = time.time() - t0
    tag = 'CORAL' if coral_weight > 0 else 'orig'
    print(f"  [{tag}] seed={seed} feat={n_features} h={h} L={n_layers} heads={n_heads} "
          f"-> Val MSE: {best_val:,.0f} ({n_params:,} params, ep {epoch+1}, {elapsed:.0f}s)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val, best_state


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Affi Pre-LN Transformer model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data directory containing train_data_neuro/ and test_dev_input/")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save trained models and normalization params")

    # ----- Model architecture hyperparameters -----
    parser.add_argument("--n_features", type=int, default=3,
                        help="Number of input features. 3 was used for the Transformer variant.")
    parser.add_argument("--h", type=int, default=384,
                        help="Transformer hidden dimension (d_model). "
                             "384 provides a good capacity-efficiency tradeoff for 239 channels.")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer encoder layers. "
                             "4 layers with h=384 gives ~6.5M params.")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads. 8 heads with h=384 -> head_dim=48.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate. 0.2 worked well for the Transformer on Affi.")

    # ----- CORAL domain adaptation -----
    parser.add_argument("--coral_weight", type=float, default=3.0,
                        help="Weight for CORAL loss. Set to 0.0 to disable domain adaptation. "
                             "3.0 was the default for CORAL-enabled models.")
    parser.add_argument("--mean_weight", type=float, default=1.0,
                        help="Weight for mean alignment loss. Set to 0.0 with coral_weight=0.0.")

    # ----- Training hyperparameters -----
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs. Transformer typically converges in ~100-150.")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size. 16 fits well for h=384 on A100 40GB.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate (after warmup). "
                             "5e-4 works well with Pre-LN Transformer + warmup.")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of linear warmup epochs before cosine decay.")
    parser.add_argument("--n_val", type=int, default=100,
                        help="Number of samples for validation split.")

    # ----- Multi-seed training -----
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated random seeds for training. "
                             "e.g., '42,123,456,789,2024' for 5-seed ensemble.")

    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load data
    print(f"\nLoading Affi data with {args.n_features} feature(s)...")
    data = load_and_prepare_data(args.data_dir, args.n_features, n_val=args.n_val)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save normalization parameters (needed for inference)
    np.savez(
        os.path.join(args.output_dir, f"normalization_affi_{args.n_features}feat.npz"),
        mean=data['mean'], std=data['std'],
    )

    # Train with each seed
    results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        tag = 'CORAL' if args.coral_weight > 0 else 'orig'
        print(f"Training seed={seed}, {tag}, n_features={args.n_features}, "
              f"h={args.h}, L={args.n_layers}, heads={args.n_heads}")
        print(f"{'='*60}")

        val_mse, state = train_one_seed(
            data=data,
            n_features=args.n_features,
            h=args.h,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            coral_weight=args.coral_weight,
            mean_weight=args.mean_weight,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            device=device,
        )
        results.append((seed, val_mse, state))

        # Save model checkpoint
        tag = 'coral' if args.coral_weight > 0 else 'orig'
        model_path = os.path.join(
            args.output_dir,
            f"model_affi_tf_{args.n_features}feat_{tag}_seed{seed}.pth",
        )
        torch.save({
            'model_state_dict': state,
            'val_mse': val_mse,
            'config': {
                'n_ch': data['n_channels'],
                'n_feat': args.n_features,
                'h': args.h,
                'n_layers': args.n_layers,
                'n_heads': args.n_heads,
                'dropout': args.dropout,
                'pre_ln': True,
                'coral_weight': args.coral_weight,
                'mean_weight': args.mean_weight,
                'seed': seed,
            },
        }, model_path)
        fsize = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Saved: {model_path} ({fsize:.1f} MB)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    for seed, val_mse, _ in results:
        print(f"  seed={seed}: Val MSE = {val_mse:,.0f}")
    best_seed, best_mse, _ = min(results, key=lambda x: x[1])
    print(f"\nBest: seed={best_seed} with Val MSE = {best_mse:,.0f}")
    print(f"Models saved to: {args.output_dir}")
