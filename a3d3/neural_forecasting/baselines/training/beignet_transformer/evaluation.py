"""
Beignet Transformer Evaluation Script
========================================
Load a trained Beignet Public Pre-LN Transformer checkpoint (h=256)
and evaluate on the validation set.

Computes denormalized MSE (original scale) so that results are directly
comparable to competition metrics.

Usage:
    python evaluation.py \
        --data_dir /path/to/train_data_neuro \
        --test_dir /path/to/test_dev_input \
        --checkpoint_path ./output/model_transformer_seed42.pth
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import TransformerForecasterPreLN
from utils import load_and_prepare_data


def evaluate(model, val_dl, std, n_val, device):
    """Run inference on the validation set and compute denormalized MSE.

    Args:
        model: Trained TransformerForecasterPreLN model in eval mode.
        val_dl: DataLoader for the validation set.
        std: Normalization standard deviation array from training.
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
    val_mse = (val_loss / n_val) * (std[..., 0] ** 2).mean()

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return val_mse, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Beignet Public Transformer on the validation set."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to train_data_neuro directory containing train_data_beignet.npz.",
    )
    parser.add_argument(
        "--test_dir", type=str, required=True,
        help="Path to test_dev_input directory containing test_data_beignet_masked.npz.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--h", type=int, default=256,
        help="Transformer hidden dimension. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_layers", type=int, default=4,
        help="Number of Transformer encoder layers. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_heads", type=int, default=4,
        help="Number of attention heads. Must match the checkpoint.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout rate. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_features", type=int, default=9,
        help="Number of input features per channel. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_val", type=int, default=100,
        help="Number of validation samples.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for evaluation.",
    )
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading Beignet data with {args.n_features} feature(s)...")
    X_tr, Y_tr, X_val, Y_val, X_target_n, mean, std = load_and_prepare_data(
        args.data_dir, args.test_dir, args.n_features, args.n_val,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Extract config from checkpoint if available, otherwise use args
    config = checkpoint.get('config', {})
    h = config.get('h', args.h)
    n_layers = config.get('n_layers', args.n_layers)
    n_heads = config.get('n_heads', args.n_heads)
    dropout = config.get('dropout', args.dropout)
    n_features = config.get('n_features', args.n_features)

    n_ch = X_val.shape[2]  # 89 for Beignet

    # Initialize model
    model = TransformerForecasterPreLN(
        n_ch=n_ch, n_feat=n_features,
        h=h, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TransformerForecasterPreLN(ch={n_ch}, feat={n_features}, "
          f"h={h}, L={n_layers}, heads={n_heads}) -> {n_params:,} params")

    # Create validation DataLoader
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_val),
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # Evaluate
    val_mse, preds, targets = evaluate(
        model, val_dl, std, len(X_val), device,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Val samples: {len(X_val)}")
    print(f"  Denormalized Val MSE: {val_mse:,.0f}")
    if 'val_mse' in checkpoint:
        print(f"  Checkpoint Val MSE:   {checkpoint['val_mse']:,.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
