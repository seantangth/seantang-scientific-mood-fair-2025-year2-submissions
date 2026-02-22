"""
Affi Transformer Evaluation Script
=====================================
Load a trained Affi Pre-LN Transformer checkpoint and evaluate on the
validation set.

Computes denormalized MSE (original scale) so that results are directly
comparable to competition metrics.

Usage:
    python evaluation.py \
        --data_dir /path/to/data \
        --checkpoint_path ./output/model_affi_tf_3feat_coral_seed42.pth

    python evaluation.py \
        --data_dir /path/to/data \
        --checkpoint_path ./output/model_affi_tf_3feat_orig_seed42.pth \
        --coral_weight 0.0
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
        description="Evaluate a trained Affi Pre-LN Transformer on the validation set."
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
        help="Number of input features. Must match the checkpoint.",
    )
    parser.add_argument(
        "--h", type=int, default=384,
        help="Transformer hidden dimension (d_model). Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_layers", type=int, default=4,
        help="Number of Transformer encoder layers. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_heads", type=int, default=8,
        help="Number of attention heads. Must match the checkpoint.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout rate. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_val", type=int, default=100,
        help="Number of validation samples.",
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
    print(f"\nLoading Affi data with {args.n_features} feature(s)...")
    data = load_and_prepare_data(args.data_dir, args.n_features, n_val=args.n_val)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Extract config from checkpoint if available, otherwise use args
    config = checkpoint.get('config', {})
    n_ch = config.get('n_ch', data['n_channels'])
    n_feat = config.get('n_feat', args.n_features)
    h = config.get('h', args.h)
    n_layers = config.get('n_layers', args.n_layers)
    n_heads = config.get('n_heads', args.n_heads)
    dropout = config.get('dropout', args.dropout)

    # Initialize model
    model = TransformerForecasterPreLN(
        n_ch=n_ch, n_feat=n_feat,
        h=h, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TransformerForecasterPreLN(ch={n_ch}, feat={n_feat}, "
          f"h={h}, L={n_layers}, heads={n_heads}) -> {n_params:,} params")

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
