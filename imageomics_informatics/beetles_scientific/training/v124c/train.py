#!/usr/bin/env python3
"""
Training script for HDR-SMood Beetles Challenge
Optimized for H100 GPU
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

from config import get_config
from data_loader import load_beetle_dataset, create_dataloaders, extract_features
from models import get_model, BioCLIP2Regressor, DINOv2Regressor
from utils import (
    set_seed,
    evaluate_spei_r2_scores,
    evaluate_all_metrics,
    EarlyStopping,
    AverageMeter,
    save_checkpoint,
    get_lr_scheduler,
    save_results,
    get_timestamp,
    save_config
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train HDR-SMood Beetles model")
    
    # Model
    parser.add_argument("--backbone", type=str, default="bioclip",
                        choices=["bioclip", "dinov2", "bioclip_meta"],
                        help="Backbone model to use")
    parser.add_argument("--unfreeze", action="store_true", default=False,
                        help="Unfreeze backbone weights (fine-tuning)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in regressor")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained model checkpoint (e.g. from stage 1)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    # Data
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token")
    
    # Mixed Precision
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    
    # Experiment
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Pre-extract features (faster training)
    parser.add_argument("--preextract_features", action="store_true", default=True,
                        help="Pre-extract backbone features before training")

    # Custom naming
    parser.add_argument("--exp_version", type=str, default="v1.0",
                        help="Experiment version tag (e.g., v1.0, v7.1)")
    
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    scaler: GradScaler = None,
    use_amp: bool = True,
    preextracted: bool = False,
    backbone: str = "bioclip"
) -> tuple:
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    all_preds = []
    all_gts = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if preextracted:
            # If pre-extracted, batch is (features, y, [optional_month])
            features = batch[0].to(device)
            y = batch[1].to(device)
            if backbone == "bioclip_meta":
                extra_feats = batch[2].to(device)
        else:
            # Normal data: x, y, month
            x = batch[0].to(device)
            y = batch[1].to(device)
            if backbone == "bioclip_meta":
                extra_feats = batch[2].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast("cuda"):
                if preextracted:
                    if backbone == "bioclip":
                        outputs = model.regressor(features)
                    elif backbone == "bioclip_meta":
                        # We need to reshape or project features if the API requires it
                        # But wait, BioCLIP2MetadataRegressor expects `forward` to take raw image + meta.
                        # It has no `forward_from_features`. We need to fix that or disable pre-extract.
                        # Actually, better to disable pre-extract for meta model for now to keep it simple.
                        # OR implement forward_from_features in BioCLIP2MetadataRegressor.
                        pass # See below for pre-extract disable logic
                    else:  # dinov2
                        outputs = model.forward_from_features(features)
                else:
                    if backbone == "bioclip_meta":
                        outputs = model(x, extra_feats)
                    else:
                        outputs = model(x)
                loss = loss_fn(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if preextracted:
                if backbone == "bioclip":
                    outputs = model.regressor(features)
                else:
                    outputs = model.forward_from_features(features)
            else:
                if backbone == "bioclip_meta":
                    outputs = model(x, extra_feats)
                else:
                    outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        loss_meter.update(loss.item(), y.size(0))
        all_preds.extend(outputs.detach().cpu().numpy().tolist())
        all_gts.extend(y.detach().cpu().numpy().tolist())
        
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
    
    # Calculate metrics
    all_gts = np.array(all_gts)
    all_preds = np.array(all_preds)
    r2_scores = evaluate_spei_r2_scores(all_gts, all_preds)
    
    return loss_meter.avg, r2_scores


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: str,
    preextracted: bool = False,
    backbone: str = "bioclip"
) -> tuple:
    """Validate model"""
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_gts = []
    
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        if preextracted:
            features = batch[0].to(device)
            y = batch[1].to(device)
            # Pre-extract not supported for bioclip_meta yet
        else:
            x = batch[0].to(device)
            y = batch[1].to(device)
            if backbone == "bioclip_meta":
                extra_feats = batch[2].to(device)
                
        if preextracted:
            if backbone == "bioclip":
                outputs = model.regressor(features)
            else:
                outputs = model.forward_from_features(features)
        else:
            if backbone == "bioclip_meta":
                outputs = model(x, extra_feats)
            else:
                outputs = model(x)
        
        loss = loss_fn(outputs, y)
        
        loss_meter.update(loss.item(), y.size(0))
        all_preds.extend(outputs.cpu().numpy().tolist())
        all_gts.extend(y.cpu().numpy().tolist())
        
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
    
    # Calculate metrics
    all_gts = np.array(all_gts)
    all_preds = np.array(all_preds)
    metrics = evaluate_all_metrics(all_gts, all_preds)
    r2_scores = (metrics["SPEI_30d"]["r2"], metrics["SPEI_1y"]["r2"], metrics["SPEI_2y"]["r2"])
    
    return loss_meter.avg, r2_scores, metrics


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Experiment name
    if args.experiment_name is None:
        # Format weight decay: 1e-5 -> wd1e5
        wd_str = f"wd{args.weight_decay:.0e}".replace("-", "")
        # Format timestamp
        timestamp = get_timestamp()
        
        # Ex: v7.1_bioclip_wd1e5_20251223_060434
        args.experiment_name = f"{args.exp_version}_{args.backbone}_{wd_str}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    # Paths
    config = get_config()
    save_dir = config.paths.model_dir / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(save_dir / "config.json", args)
    
    log_dir = config.paths.log_dir / args.experiment_name
    writer = SummaryWriter(log_dir)
    
    # Load dataset
    print("Loading dataset...")
    train_dset, val_dset, transform = load_beetle_dataset(
        hf_token=args.hf_token,
        backbone=args.backbone
    )
    print(f"Train samples: {len(train_dset)}")
    print(f"Val samples: {len(val_dset)}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dset, val_dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Determine freeze status
    freeze_backbone = not args.unfreeze
    
    # Create model
    print(f"\nCreating {args.backbone} model...")
    print(f"Backbone frozen: {freeze_backbone}")
    
    model = get_model(
        backbone=args.backbone,
        dropout=args.dropout,
        freeze_backbone=freeze_backbone,
        device=device
    )
    
    # Load pretrained weights if specified
    if args.pretrained_path:
        print(f"Loading pretrained weights from: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Handle case where we load a frozen model (only head) into a full model
        # or vice versa. We use strict=False to allow flexibility.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print(f"  First few missing: {missing[:5]}")
    
    # Pre-extract features if requested
    # Only possible if backbone is frozen AND not bioclip_meta (which needs complex dataloader handling)
    if args.preextract_features and freeze_backbone and args.backbone != "bioclip_meta":
        print("\nPre-extracting features (this may take a while)...")
        
        # Get backbone
        if args.backbone == "bioclip":
            backbone_model = model.bioclip
        else:
            backbone_model = model.dino
        
        # Use GPU feature store if possible (H100 has 80GB, dataset features are small)
        # 25k images * 768 float32 * 4 bytes ~= 75MB. This is tiny.
        # We can keep everything on GPU.
        
        train_X, train_Y = extract_features(
            train_loader, backbone_model, args.backbone, device
        )
        val_X, val_Y = extract_features(
            val_loader, backbone_model, args.backbone, device
        )
        
        # Move to GPU immediately for blazing fast training
        print("Moving features to GPU memory...")
        train_X = train_X.to(device)
        train_Y = train_Y.to(device)
        val_X = val_X.to(device)
        val_Y = val_Y.to(device)
        
        # Create new dataloaders with extracted features on GPU
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_X, train_Y),
            batch_size=args.batch_size, # Can use larger batch size for features
            shuffle=True,
            num_workers=0 # No workers needed for GPU tensors
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_X, val_Y),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        preextracted = True
        print("Feature extraction complete! Features cached on GPU.")
    else:
        preextracted = False
    
    # Optimizer - only train regressor if backbone is frozen
    if freeze_backbone:
        if args.backbone == "bioclip":
            params = model.regressor.parameters()
        elif args.backbone == "bioclip_meta":
            params = list(model.meta_encoder.parameters()) + list(model.regressor.parameters())
        else:
            params = list(model.tokens_to_linear.parameters()) + list(model.regressor.parameters())
    else:
        params = model.parameters()
    
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type="cosine",
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs
    )
    
    # Loss function
    loss_fn = nn.MSELoss()
    
    # Mixed precision
    scaler = GradScaler() if args.amp else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    best_r2 = -float("inf")
    best_epoch = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_r2 = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scaler, args.amp, preextracted, args.backbone
        )
        
        # Validate
        val_loss, val_r2, val_metrics = validate(
            model, val_loader, loss_fn, device,
            preextracted, args.backbone
        )
        
        # Step scheduler
        scheduler.step()
        
        # Calculate average R2
        avg_train_r2 = sum(train_r2) / 3
        avg_val_r2 = sum(val_r2) / 3
        
        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("R2/train_avg", avg_train_r2, epoch)
        writer.add_scalar("R2/val_avg", avg_val_r2, epoch)
        writer.add_scalar("R2/val_SPEI_30d", val_r2[0], epoch)
        writer.add_scalar("R2/val_SPEI_1y", val_r2[1], epoch)
        writer.add_scalar("R2/val_SPEI_2y", val_r2[2], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f} | Train R²: {avg_train_r2:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val R²: {avg_val_r2:.4f}")
        print(f"  Val R² (30d/1y/2y): {val_r2[0]:.4f} / {val_r2[1]:.4f} / {val_r2[2]:.4f}")
        
        # Save best model
        is_best = avg_val_r2 > best_r2
        if is_best:
            best_r2 = avg_val_r2
            best_epoch = epoch
            print(f"  *** New best model! ***")
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            save_dir, is_best, score=avg_val_r2
        )
        
        print(f"  Best R²: {best_r2:.4f} (epoch {best_epoch+1})")
        
        # Early stopping
        if early_stopping(avg_val_r2):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final results
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation R²: {best_r2:.4f} (epoch {best_epoch+1})")
    print(f"Model saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    # Save final results
    save_results(save_dir / "results.json", val_metrics)
    
    writer.close()


if __name__ == "__main__":
    main()
