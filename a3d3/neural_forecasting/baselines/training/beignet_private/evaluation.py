"""
Beignet Private TCN Evaluation Script
========================================
Load a trained Beignet Private TCN checkpoint (h=256) and evaluate on a
held-out validation split from the private domain data.

Each private domain (priv1: 2022-06-01, priv2: 2022-06-02) is evaluated
independently. The validation split is 1/6 of the data (random, seeded).

Computes denormalized MSE (original scale) so that results are directly
comparable to competition metrics.

Usage:
    # Evaluate priv1 model
    python evaluation.py \
        --data_dir /path/to/train_data_neuro \
        --checkpoint_path ./output/model_tcn_priv1_seed42.pth \
        --domain priv1

    # Evaluate priv2 model
    python evaluation.py \
        --data_dir /path/to/train_data_neuro \
        --checkpoint_path ./output/model_tcn_priv2_seed42.pth \
        --domain priv2
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import TCNForecaster
from utils import load_private_data


def evaluate(model, val_dl, domain_std, n_val, device):
    """Run inference on the validation set and compute denormalized MSE.

    Args:
        model: Trained TCNForecaster model in eval mode.
        val_dl: DataLoader for the validation set.
        domain_std: Combined normalization std array.
        n_val: Number of validation samples.
        device: Torch device.

    Returns:
        val_mse: Denormalized validation MSE (original scale).
        all_preds: Numpy array of normalized predictions.
        all_targets: Numpy array of normalized targets.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += ((pred - yb) ** 2).sum().item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    # Denormalized MSE
    val_mse = (val_loss / n_val) * (domain_std[..., 0] ** 2).mean()

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return val_mse, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Beignet Private TCN on a validation split."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to train_data_neuro directory containing the private .npz files.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--domain", type=str, required=True, choices=["priv1", "priv2"],
        help="Which private domain to evaluate: 'priv1' (2022-06-01) or 'priv2' (2022-06-02).",
    )
    parser.add_argument(
        "--n_feat", type=int, default=1,
        help="Number of input features. Must match the checkpoint.",
    )
    parser.add_argument(
        "--h_size", type=int, default=128,
        help="TCN hidden dimension. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_layers", type=int, default=3,
        help="Number of TCN layers. Must match the checkpoint.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate. Must match the checkpoint.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val split (must match training).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for evaluation.",
    )
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading Beignet private data...")
    train_priv1, train_priv2, mean_priv, std_priv = load_private_data(
        args.data_dir, args.n_feat,
    )

    # Select the appropriate domain
    if args.domain == "priv1":
        train_data = train_priv1
        print(f"Evaluating on Priv1 (2022-06-01): {len(train_data)} samples")
    else:
        train_data = train_priv2
        print(f"Evaluating on Priv2 (2022-06-02): {len(train_data)} samples")

    # Reproduce the same train/val split as training
    np.random.seed(args.seed)
    n_feat = args.n_feat
    X_all = train_data[:, :10, :, :n_feat].astype(np.float32)
    Y_all = train_data[:, 10:, :, 0].astype(np.float32)

    dm = mean_priv[..., :n_feat]
    ds = std_priv[..., :n_feat]
    X_all_n = (X_all - dm) / ds
    Y_all_n = (Y_all - mean_priv[..., 0]) / std_priv[..., 0]

    n = len(X_all_n)
    idx = np.random.permutation(n)
    n_val = max(8, n // 6)
    X_val = X_all_n[idx[:n_val]]
    Y_val = Y_all_n[idx[:n_val]]

    print(f"Val samples: {n_val} (seed={args.seed})")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Extract config from checkpoint if available, otherwise use args
    config = checkpoint.get('config', {})
    n_feat = config.get('n_feat', args.n_feat)
    h_size = config.get('h_size', args.h_size)
    n_layers = config.get('n_layers', args.n_layers)
    dropout = config.get('dropout', args.dropout)

    n_ch = X_all.shape[2]  # 89

    # Initialize model
    model = TCNForecaster(
        n_ch=n_ch, n_feat=n_feat,
        h=h_size, n_layers=n_layers, dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TCNForecaster(ch={n_ch}, feat={n_feat}, "
          f"h={h_size}, L={n_layers}) -> {n_params:,} params")

    # Create validation DataLoader
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_val),
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # Evaluate
    val_mse, preds, targets = evaluate(
        model, val_dl, std_priv, n_val, device,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.domain})")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Domain: {args.domain}")
    print(f"  Val samples: {n_val}")
    print(f"  Denormalized Val MSE: {val_mse:,.0f}")
    if 'val_mse' in checkpoint:
        print(f"  Checkpoint Val MSE:   {checkpoint['val_mse']:,.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
