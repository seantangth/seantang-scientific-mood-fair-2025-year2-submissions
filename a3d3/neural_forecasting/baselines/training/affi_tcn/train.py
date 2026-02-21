"""
Affi TCN Training Script
=========================
Train a Temporal Convolutional Network (TCN) on Monkey A (Affi) neural data
with optional CORAL domain adaptation loss.

This script supports two configurations that were used in the final ensemble:
    - 3-feature model (d=384, 4 layers): Uses features 0, 1, 2 from the raw data
    - 1-feature model (d=384, 6 layers): Uses only feature 0 (primary neural signal)

The CORAL (CORrelation ALignment) loss aligns the second-order statistics of
source (training) and target (test) domain feature distributions, improving
generalization to unseen recording sessions.

Usage:
    # Train 3-feature model with CORAL
    python train.py --n_features 3 --n_layers 4 --coral_weight 3.0

    # Train 1-feature model without CORAL
    python train.py --n_features 1 --n_layers 6 --coral_weight 0.0

    # Train with multiple seeds for ensembling
    python train.py --n_features 3 --n_layers 4 --seeds 42,123,456,789,2024
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

from model import TCNForecasterLarge


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

def load_and_prepare_data(data_dir: str, n_features: int, n_val: int = 100):
    """Load Affi neural data and prepare train/val/target splits.

    The data is z-score normalized per channel using statistics computed
    from the training set input window (first 10 timesteps).

    Args:
        data_dir: Path to the directory containing train_data_neuro/ and
            test_dev_input/ subdirectories.
        n_features: Number of features to use (1 or 3).
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
    n_features: int,
    hidden_size: int = 384,
    n_layers: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.1,
    coral_weight: float = 3.0,
    mean_weight: float = 1.0,
    seed: int = 42,
    epochs: int = 500,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    warmup_epochs: int = 10,
    use_augmentation: bool = True,
    device: torch.device = None,
):
    """Train a single TCN model with a given random seed.

    Args:
        data: Dictionary from load_and_prepare_data().
        n_features: Number of input features (1 or 3).
        hidden_size: TCN hidden dimension.
        n_layers: Number of TCN blocks.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        coral_weight: Weight for CORAL loss (0 to disable domain adaptation).
        mean_weight: Weight for mean alignment loss.
        seed: Random seed for reproducibility.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        batch_size: Training batch size.
        lr: Peak learning rate (after warmup).
        warmup_epochs: Number of linear warmup epochs.
        use_augmentation: Whether to apply data augmentation.
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
    model = TCNForecasterLarge(
        n_channels=n_channels,
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=n_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: TCNForecasterLarge(ch={n_channels}, feat={n_features}, "
          f"h={hidden_size}, L={n_layers}, k={kernel_size}) -> {n_params:,} params")

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
            if use_augmentation:
                xb, yb = augment_batch(xb, yb)

            optimizer.zero_grad()

            with autocast():
                # Forward pass on source (training) data
                pred = model(xb)
                mse_loss = F.mse_loss(pred, yb)

                # CORAL domain adaptation loss
                if coral_weight > 0:
                    # Extract features for domain adaptation
                    # Re-run forward to get intermediate features
                    # For TCN, we use the prediction as proxy features
                    pred_tgt = model(xt)
                    # Flatten predictions as feature vectors for CORAL
                    feat_src = pred.reshape(pred.size(0), -1).float()
                    feat_tgt = pred_tgt.reshape(pred_tgt.size(0), -1).float()
                    c_loss = coral_loss(feat_src, feat_tgt)
                    m_loss = mean_alignment_loss(feat_src, feat_tgt)
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
                    pred = model(xb)
                    val_loss += F.mse_loss(pred, yb, reduction='sum').item()

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
    print(f"  [{tag}] seed={seed} -> Best Val MSE: {best_val:,.0f} "
          f"({n_params:,} params, {epoch+1} epochs, {elapsed:.0f}s)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val, best_state


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Affi TCN model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data directory containing train_data_neuro/ and test_dev_input/")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save trained models and normalization params")

    # ----- Model architecture hyperparameters -----
    parser.add_argument("--n_features", type=int, default=3,
                        help="Number of input features (1 or 3). "
                             "3-feat uses features [0,1,2]; 1-feat uses only feature [0].")
    parser.add_argument("--hidden_size", type=int, default=384,
                        help="Hidden dimension for TCN layers and attention. "
                             "384 was used for both Affi configs.")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of TCN blocks. "
                             "4 layers for 3-feat, 6 layers for 1-feat in final ensemble.")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Convolution kernel size for causal convolutions.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate. 0.1 worked well for Affi (large dataset).")

    # ----- CORAL domain adaptation -----
    parser.add_argument("--coral_weight", type=float, default=3.0,
                        help="Weight for CORAL loss. Set to 0.0 to disable domain adaptation. "
                             "3.0 was the default for CORAL-enabled models.")
    parser.add_argument("--mean_weight", type=float, default=1.0,
                        help="Weight for mean alignment loss. Set to 0.0 with coral_weight=0.0.")

    # ----- Training hyperparameters -----
    parser.add_argument("--epochs", type=int, default=500,
                        help="Maximum training epochs. Training often stops early (~150-200).")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size. 32 fits well on a single GPU.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate (after warmup). "
                             "1e-4 was used for TCN training.")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of linear warmup epochs before cosine decay.")
    parser.add_argument("--n_val", type=int, default=100,
                        help="Number of samples for validation split.")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable data augmentation.")

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
        print(f"Training seed={seed}, n_features={args.n_features}, "
              f"n_layers={args.n_layers}, CORAL={'ON' if args.coral_weight > 0 else 'OFF'}")
        print(f"{'='*60}")

        val_mse, state = train_one_seed(
            data=data,
            n_features=args.n_features,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            coral_weight=args.coral_weight,
            mean_weight=args.mean_weight,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            use_augmentation=not args.no_augmentation,
            device=device,
        )
        results.append((seed, val_mse, state))

        # Save model checkpoint
        tag = 'coral' if args.coral_weight > 0 else 'orig'
        model_path = os.path.join(
            args.output_dir,
            f"model_affi_tcn_{args.n_features}feat_{tag}_seed{seed}.pth",
        )
        torch.save({
            'model_state_dict': state,
            'val_mse': val_mse,
            'config': {
                'n_channels': data['n_channels'],
                'n_features': args.n_features,
                'hidden_size': args.hidden_size,
                'n_layers': args.n_layers,
                'kernel_size': args.kernel_size,
                'dropout': args.dropout,
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
