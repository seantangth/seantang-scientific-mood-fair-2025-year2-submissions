#!/usr/bin/env python3
"""
5-Fold Cross-Validation Training Script for HDR-SMood Beetles Challenge
Optimized for H100 GPU
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from pathlib import Path
import numpy as np

from config import get_config
from data_loader import load_beetle_dataset
from models import get_model
from utils import (
    set_seed,
    evaluate_all_metrics,
    EarlyStopping,
    AverageMeter,
    save_checkpoint,
    get_lr_scheduler,
    save_results,
    get_timestamp,
    save_config
)
# Reuse training logic from train.py
from train import train_one_epoch, validate

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train HDR-SMood Beetles model with K-Fold CV")
    
    # K-Fold specific
    parser.add_argument("--k_folds", type=int, default=5,
                        help="Number of folds for Cross-Validation")
    
    # Model
    parser.add_argument("--backbone", type=str, default="dinov2",
                        choices=["bioclip", "dinov2", "bioclip_meta"],
                        help="Backbone model to use")
    parser.add_argument("--unfreeze", action="store_true", default=False,
                        help="Unfreeze backbone weights (fine-tuning)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in regressor")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained model checkpoint (e.g. from stage 1)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=15,
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
                        help="Experiment name prefix")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # No pre-extract for K-Fold usually if we do Unfreeze
    # If Freeze + Pre-extract is needed, we'd implement it per fold. 
    # For now assuming Unfreeze (H100 power) or standard Freeze training.
    
    parser.add_argument("--exp_version", type=str, default="v6.0",
                        help="Experiment version tag")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Experiment name
    if args.experiment_name is None:
        timestamp = get_timestamp()
        args.experiment_name = f"{args.exp_version}_kfold_{args.backbone}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"K-Fold Experiment: {args.experiment_name}")
    print(f"Folds: {args.k_folds}")
    print(f"Backbone: {args.backbone}")
    print(f"Unfreeze: {args.unfreeze}")
    print(f"{'='*60}\n")
    
    # Paths
    config = get_config()
    base_save_dir = config.paths.model_dir / args.experiment_name
    base_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(base_save_dir / "config.json", args)
    
    # Load dataset
    print("Loading dataset...")
    # We load both splits and concatenate them
    train_dset_part, val_dset_part, transform = load_beetle_dataset(
        hf_token=args.hf_token,
        backbone=args.backbone
    )
    full_dataset = ConcatDataset([train_dset_part, val_dset_part])
    print(f"Total samples: {len(full_dataset)}")
    
    # K-Fold Split
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # Store results
    fold_results = []
    
    # We need to access collate_fn from create_dataloaders logic or import it?
    # It's inside create_dataloaders. Let's look at data_loader.py again.
    # get_collate_fn is exported!
    from data_loader import get_collate_fn
    collate_fn = get_collate_fn()
    
    # Iterate Folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'-'*40}")
        print(f"FOLD {fold+1}/{args.k_folds}")
        print(f"{'-'*40}")
        
        # Setup directories for this fold
        fold_save_dir = base_save_dir / f"fold_{fold+1}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)
        
        fold_log_dir = config.paths.log_dir / args.experiment_name / f"fold_{fold+1}"
        writer = SummaryWriter(fold_log_dir)
        
        # Create Subsets
        train_sub = Subset(full_dataset, train_ids)
        val_sub = Subset(full_dataset, val_ids)
        
        print(f"Train samples: {len(train_sub)}")
        print(f"Val samples: {len(val_sub)}")
        
        # DataLoaders
        train_loader = DataLoader(
            train_sub,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Initialize Model (Fresh for each fold)
        freeze_backbone = not args.unfreeze
        model = get_model(
            backbone=args.backbone,
            dropout=args.dropout,
            freeze_backbone=freeze_backbone,
            device=device
        )
        
        # Init weights (if pretrained_path provided)
        # Assuming pretrained_path is a Stage 1 model (general best), we load it as starting point
        if args.pretrained_path:
            print(f"Loading pretrained weights from: {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location=device)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        
        # Optimizer
        if freeze_backbone:
            if args.backbone == "bioclip":
                params = model.regressor.parameters()
            elif args.backbone == "bioclip_meta":
                params = list(model.meta_encoder.parameters()) + list(model.regressor.parameters())
            else:
                params = list(model.tokens_to_linear.parameters()) + list(model.regressor.parameters())
        else:
            params = model.parameters() # Train everything
            
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        
        # Scheduler
        scheduler = get_lr_scheduler(
            optimizer,
            scheduler_type="cosine",
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs
        )
        
        loss_fn = nn.MSELoss()
        scaler = GradScaler() if args.amp else None
        early_stopping = EarlyStopping(patience=args.patience)
        
        # Training Loop
        best_r2 = -float("inf")
        best_epoch = 0
        
        for epoch in range(args.epochs):
            # Reuse train_one_epoch from train.py
            train_loss, train_r2 = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device,
                scaler, args.amp, preextracted=False, backbone=args.backbone
            )
            
            val_loss, val_r2, val_metrics = validate(
                model, val_loader, loss_fn, device,
                preextracted=False, backbone=args.backbone
            )
            
            scheduler.step()
            
            avg_val_r2 = sum(val_r2) / 3
            
            # Logging
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("R2/val_avg", avg_val_r2, epoch)
            
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val R²: {avg_val_r2:.4f}")
            
            # Save Best
            is_best = avg_val_r2 > best_r2
            if is_best:
                best_r2 = avg_val_r2
                best_epoch = epoch
                # Save fold best (symlink or copy logic handled by save_checkpoint, but we want unique names)
                # save_checkpoint saves to 'best.pth' and 'best_model...'. 
                # reliable way:
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    fold_save_dir, is_best, score=avg_val_r2
                )
                
            if early_stopping(avg_val_r2):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Fold {fold+1} Best R²: {best_r2:.4f}")
        fold_results.append(best_r2)
        writer.close()
        
        # Cleanup to save memory
        del model, optimizer, scaler, train_loader, val_loader
        torch.cuda.empty_cache()
        
    # Summary
    print(f"\n{'='*60}")
    print("K-Fold Cross-Validation Complete")
    print(f"Folds: {args.k_folds}")
    print(f"Results: {fold_results}")
    print(f"Average R²: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    print(f"{'='*60}\n")
    
    # Save overall summary
    import json
    with open(base_save_dir / "summary.json", "w") as f:
        json.dump({
            "folds": fold_results,
            "mean": float(np.mean(fold_results)),
            "std": float(np.std(fold_results))
        }, f, indent=4)

if __name__ == "__main__":
    main()
