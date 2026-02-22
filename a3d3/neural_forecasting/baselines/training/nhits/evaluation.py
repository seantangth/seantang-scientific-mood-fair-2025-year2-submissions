"""
NHiTS Evaluation Script
=========================
Load a trained NHiTS checkpoint and evaluate on the validation set.

NHiTS uses 1 feature from the Beignet dataset with CORAL domain adaptation.
Computes denormalized MSE (original scale) so that results are directly
comparable to competition metrics.

Usage:
    python evaluation.py \
        --train_dir /path/to/train_data_neuro \
        --test_dir /path/to/test_dev_input \
        --checkpoint_path ./output/model_nhits_coral.pth
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import NHiTSForecaster
from utils import load_and_prepare_data


def evaluate(model, val_dl, std, n_val, device):
    """Run inference on the validation set and compute denormalized MSE.

    Args:
        model: Trained NHiTSForecaster model in eval mode.
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
        description="Evaluate a trained NHiTS model on the validation set."
    )
    parser.add_argument(
        "--train_dir", type=str, required=True,
        help="Directory containing train_data_beignet.npz.",
    )
    parser.add_argument(
        "--test_dir", type=str, required=True,
        help="Directory containing test_data_beignet_masked.npz.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256,
        help="NHiTS hidden dimension. Must match the checkpoint.",
    )
    parser.add_argument(
        "--n_stacks", type=int, default=2,
        help="Number of NHiTS stacks. Must match the checkpoint.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2,
        help="Number of blocks per stack. Must match the checkpoint.",
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
    print(f"\nLoading Beignet data (1 feature for NHiTS)...")
    data = load_and_prepare_data(args.train_dir, args.test_dir, n_val=args.n_val)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    n_channels = data['n_channels']

    # Initialize model
    model = NHiTSForecaster(
        n_channels=n_channels,
        n_features=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        n_stacks=args.n_stacks,
        seq_len=10,
        dropout=args.dropout,
        output_steps=10,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NHiTS parameters: {n_params:,}")

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
