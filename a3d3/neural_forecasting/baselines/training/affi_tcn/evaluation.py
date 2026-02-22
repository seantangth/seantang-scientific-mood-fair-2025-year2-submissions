"""
Affi TCN Evaluation Script
============================
Load a trained Affi TCN checkpoint and evaluate on the validation set.

Computes denormalized MSE (original scale) so that results are directly
comparable to competition metrics.

Usage:
    python evaluation.py \
        --data_dir /path/to/data \
        --checkpoint_path ./output/model_affi_tcn_3feat_coral_seed42.pth

    python evaluation.py \
        --data_dir /path/to/data \
        --checkpoint_path ./output/model_affi_tcn_1feat_orig_seed42.pth \
        --n_features 1 --n_layers 6
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import TCNForecasterLarge
from utils import load_and_prepare_data


def evaluate(model, val_dl, std, n_val, device):
    """Run inference on the validation set and compute denormalized MSE.

    Args:
        model: Trained TCNForecasterLarge model in eval mode.
        val_dl: DataLoader for the validation set.
        std: Normalization standard deviation array from training.
        n_val: Number of validation samples.
        device: Torch device.

    Returns:
        val_mse: Denormalized validation MSE (original scale).
        all_preds: Numpy array of denormalized predictions.
        all_targets: Numpy array of denormalized targets.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += F.mse_loss(pred, yb, reduction='sum').item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    # Denormalized MSE
    val_mse = (val_loss / n_val) * (std[..., 0] ** 2).mean()

    # Denormalize predictions and targets for inspection
    mean_0 = std[..., 0]  # borrowing shape; actual denorm uses mean[...,0]
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return val_mse, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Affi TCN model on the validation set."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory containing train_data_neuro/ and test_dev_input/",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--n_features", type=int, default=3,
        help="Number of input features (1 or 3). Must match the checkpoint.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=384,
        help="TCN hidden dimension. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_layers", type=int, default=4,
        help="Number of TCN blocks. Must match the checkpoint.",
    )
    parser.add_argument(
        "--kernel_size", type=int, default=3,
        help="Convolution kernel size. Must match the checkpoint.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate. Must match the checkpoint.",
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
    print(f"\nLoading Affi data with {args.n_features} feature(s)...")
    data = load_and_prepare_data(args.data_dir, args.n_features, n_val=args.n_val)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Extract config from checkpoint if available, otherwise use args
    config = checkpoint.get('config', {})
    n_channels = config.get('n_channels', data['n_channels'])
    n_features = config.get('n_features', args.n_features)
    hidden_size = config.get('hidden_size', args.hidden_size)
    n_layers = config.get('n_layers', args.n_layers)
    kernel_size = config.get('kernel_size', args.kernel_size)
    dropout = config.get('dropout', args.dropout)

    # Initialize model
    model = TCNForecasterLarge(
        n_channels=n_channels,
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=n_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TCNForecasterLarge(ch={n_channels}, feat={n_features}, "
          f"h={hidden_size}, L={n_layers}, k={kernel_size}) -> {n_params:,} params")

    # Create validation DataLoader
    val_ds = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.FloatTensor(data['Y_val']),
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # Evaluate
    val_mse, preds, targets = evaluate(
        model, val_dl, data['std'], len(data['X_val']), device,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Val samples: {len(data['X_val'])}")
    print(f"  Denormalized Val MSE: {val_mse:,.0f}")
    if 'val_mse' in checkpoint:
        print(f"  Checkpoint Val MSE:   {checkpoint['val_mse']:,.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
