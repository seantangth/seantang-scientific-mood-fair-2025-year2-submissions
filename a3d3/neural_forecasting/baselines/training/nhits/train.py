"""
Train NHiTS with CORAL domain adaptation (v192d).

NHiTS uses only 1 feature (feature 0) from the Beignet dataset.
CORAL aligns hidden-layer feature distributions between the labelled
training domain and the unlabelled test (target) domain.

Usage:
    python train.py --train_dir /path/to/train_data_neuro \
                    --test_dir  /path/to/test_dev_input \
                    --output_dir /path/to/output
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import NHiTSForecaster


# ---------------------------------------------------------------------------
# CORAL loss
# ---------------------------------------------------------------------------

def coral_loss(source, target):
    """CORAL: minimize difference of second-order statistics (covariance)."""
    d = source.size(1)
    cs = (source - source.mean(0, keepdim=True)).T @ \
         (source - source.mean(0, keepdim=True)) / (source.size(0) - 1 + 1e-8)
    ct = (target - target.mean(0, keepdim=True)).T @ \
         (target - target.mean(0, keepdim=True)) / (target.size(0) - 1 + 1e-8)
    return ((cs - ct) ** 2).sum() / (4 * d)


def mean_alignment_loss(source, target):
    """Align first-order statistics (mean) between domains."""
    return ((source.mean(0) - target.mean(0)) ** 2).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_dl, val_dl, target_dl, optimizer, scheduler,
          device, epochs, patience, coral_weight, mean_weight,
          X_val, std):
    """Train with MSE + CORAL + mean-alignment losses."""
    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_mse, train_coral, n_batches = 0.0, 0.0, 0
        target_iter = iter(target_dl)

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            # Get a target-domain batch (cycle if exhausted)
            try:
                (xt,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_dl)
                (xt,) = next(target_iter)
            xt = xt.to(device)

            optimizer.zero_grad()
            pred, feat_src = model(xb, return_features=True)
            _, feat_tgt = model(xt, return_features=True)

            mse = ((pred - yb) ** 2).mean()
            coral = coral_loss(feat_src, feat_tgt)
            mean_a = mean_alignment_loss(feat_src, feat_tgt)
            loss = mse + coral_weight * coral + mean_weight * mean_a

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_mse += mse.item()
            train_coral += coral.item()
            n_batches += 1

        scheduler.step()
        train_mse /= n_batches
        train_coral /= n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += ((model(xb) - yb) ** 2).sum().item()
        val_mse = (val_loss / len(X_val)) * (std[..., 0] ** 2).mean()

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f'Epoch {epoch+1:3d}: MSE={train_mse:.4f}, '
                  f'CORAL={train_coral:.6f}, Val={val_mse:.0f} ***')
        else:
            no_improve += 1
            if epoch % 20 == 0:
                print(f'Epoch {epoch+1:3d}: MSE={train_mse:.4f}, '
                      f'CORAL={train_coral:.6f}, Val={val_mse:.0f}')

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print(f'\nBest Val MSE: {best_val:.0f}')
    return best_state, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train NHiTS with CORAL')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing train_data_beignet.npz')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test_data_beignet_masked.npz')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save model checkpoint')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_stacks', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--coral_weight', type=float, default=1.0)
    parser.add_argument('--mean_weight', type=float, default=0.5)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ------------------------------------------------------------------
    # Data: NHiTS uses only feature 0
    # ------------------------------------------------------------------
    train_data = np.load(
        os.path.join(args.train_dir, 'train_data_beignet.npz'))['arr_0']
    test_public = np.load(
        os.path.join(args.test_dir, 'test_data_beignet_masked.npz'))['arr_0']

    X_train = train_data[:, :10, :, 0:1].astype(np.float32)  # 1 feature
    Y_train = train_data[:, 10:, :, 0].astype(np.float32)
    X_target = test_public[:, :10, :, 0:1].astype(np.float32)

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train_norm = (X_train - mean) / std
    Y_train_norm = (Y_train - mean[..., 0]) / std[..., 0]
    X_target_norm = (X_target - mean) / std

    n_val = args.n_val
    X_tr, X_val = X_train_norm[:-n_val], X_train_norm[-n_val:]
    Y_tr, Y_val = Y_train_norm[:-n_val], Y_train_norm[-n_val:]

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    target_ds = TensorDataset(torch.FloatTensor(X_target_norm))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    target_dl = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True)

    n_channels = X_train.shape[2]
    print(f'Train: {len(X_tr)}, Val: {len(X_val)}, '
          f'Target: {len(X_target_norm)}, Channels: {n_channels}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
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
    print(f'NHiTS parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f'Training NHiTS with CORAL={args.coral_weight}, '
          f'Mean={args.mean_weight}')
    print('-' * 70)

    best_state, best_val = train(
        model, train_dl, val_dl, target_dl, optimizer, scheduler,
        device, args.epochs, args.patience, args.coral_weight,
        args.mean_weight, X_val, std,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    torch.save({
        'model_state_dict': best_state,
        'val_mse': best_val,
    }, os.path.join(args.output_dir, 'model_nhits_coral.pth'))

    np.savez(
        os.path.join(args.output_dir, 'normalization_beignet_nhits.npz'),
        mean=mean, std=std,
    )

    print(f'Saved to {args.output_dir}')


if __name__ == '__main__':
    main()
