#!/usr/bin/env python3
"""
Evaluation script for V124c MLP LP-FT model (HDR-SMood Beetles Challenge)

Usage:
    python evaluation.py --model_path <path_to_checkpoint> --backbone bioclip

This script loads a trained model checkpoint and evaluates it on the
validation split of the sentinel-beetles dataset, reporting R², MAE,
and RMSE for each SPEI target (SPEI_30d, SPEI_1y, SPEI_2y).
"""
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_config
from data_loader import load_beetle_dataset, get_collate_fn
from models import get_model
from utils import evaluate_all_metrics, compile_event_predictions, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HDR-SMood Beetles model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--backbone", type=str, default="bioclip",
                        choices=["bioclip", "dinov2", "bioclip_meta"],
                        help="Backbone model used during training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, backbone="bioclip"):
    """Evaluate model on dataloader, return metrics."""
    model.eval()
    all_preds = []
    all_gts = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        x = batch[0].to(device)
        y = batch[1]  # SPEI targets

        if backbone == "bioclip_meta":
            extra_feats = batch[2].to(device)
            outputs = model(x, extra_feats)
        else:
            outputs = model(x)

        all_preds.extend(outputs.cpu().numpy().tolist())
        all_gts.extend(y.numpy().tolist())

    all_gts = np.array(all_gts)
    all_preds = np.array(all_preds)

    metrics = evaluate_all_metrics(all_gts, all_preds)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset (validation split)
    print("Loading validation dataset...")
    _, val_dset, transform = load_beetle_dataset(
        hf_token=args.hf_token,
        backbone=args.backbone
    )

    collate_fn = get_collate_fn()
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    print(f"Creating {args.backbone} model...")
    model = get_model(
        backbone=args.backbone,
        freeze_backbone=True,
        device=device
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Evaluate
    metrics = evaluate(model, val_loader, device, args.backbone)

    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    for target in ["SPEI_30d", "SPEI_1y", "SPEI_2y"]:
        m = metrics[target]
        print(f"  {target}: R²={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
    avg = metrics["average"]
    print(f"  Average: R²={avg['r2']:.4f}  MAE={avg['mae']:.4f}  RMSE={avg['rmse']:.4f}")
    print(f"{'='*60}")

    # Save results
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
