"""
Utility functions for HDR-SMood Beetles Challenge
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import random
import argparse
import sys
import zipfile
import shutil


class Logger:
    """
    Logger that writes to both console and file simultaneously.
    Usage:
        logger = Logger(log_path)
        logger.log("message")  # or just print() after setup_global_logging()
    """
    def __init__(self, log_path: Path, mode: str = "w"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, mode, encoding="utf-8")
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def log(self, message: str):
        """Explicit log method with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        self.write(formatted)


def setup_logging(log_path: Path) -> Logger:
    """
    Setup global logging to both console and file.
    All print() statements will be captured.

    Args:
        log_path: Path to log file

    Returns:
        Logger instance
    """
    logger = Logger(log_path)
    sys.stdout = logger
    return logger


def teardown_logging(logger: Logger):
    """Restore stdout and close log file"""
    sys.stdout = logger.terminal
    logger.close()


def create_zip_archive(
    source_dir: Path,
    output_path: Optional[Path] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> Path:
    """
    Create a zip archive of a directory.

    Args:
        source_dir: Directory to compress
        output_path: Output zip path (default: source_dir.zip)
        include_patterns: Only include files matching these patterns (e.g., ["*.pth", "*.json"])
        exclude_patterns: Exclude files matching these patterns (e.g., ["*.log"])

    Returns:
        Path to created zip file
    """
    source_dir = Path(source_dir)

    if output_path is None:
        output_path = source_dir.parent / f"{source_dir.name}.zip"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating archive: {output_path}")
    print(f"Source directory: {source_dir}")

    file_count = 0
    total_size = 0

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                # Check include patterns
                if include_patterns:
                    if not any(file_path.match(p) for p in include_patterns):
                        continue

                # Check exclude patterns
                if exclude_patterns:
                    if any(file_path.match(p) for p in exclude_patterns):
                        continue

                # Add to archive
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)
                file_count += 1
                total_size += file_path.stat().st_size

    zip_size = output_path.stat().st_size
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0

    print(f"Archive created successfully!")
    print(f"  Files: {file_count}")
    print(f"  Original size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Compressed size: {zip_size / 1024 / 1024:.1f} MB")
    print(f"  Compression ratio: {compression_ratio:.1f}%")

    return output_path


def create_experiment_archive(
    model_dir: Path,
    log_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    experiment_name: str = "experiment"
) -> Path:
    """
    Create a complete archive of an experiment including models and logs.

    Args:
        model_dir: Directory containing model checkpoints
        log_dir: Directory containing TensorBoard logs (optional)
        output_dir: Where to save the zip file
        experiment_name: Name for the archive

    Returns:
        Path to created zip file
    """
    model_dir = Path(model_dir)

    if output_dir is None:
        output_dir = model_dir.parent

    output_path = Path(output_dir) / f"{experiment_name}.zip"

    print(f"\n{'='*50}")
    print(f"Creating experiment archive: {experiment_name}")
    print(f"{'='*50}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model files
        if model_dir.exists():
            print(f"Adding models from: {model_dir}")
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    arcname = Path("models") / file_path.relative_to(model_dir)
                    zipf.write(file_path, arcname)

        # Add log files
        if log_dir and Path(log_dir).exists():
            print(f"Adding logs from: {log_dir}")
            for file_path in Path(log_dir).rglob("*"):
                if file_path.is_file():
                    arcname = Path("logs") / file_path.relative_to(log_dir)
                    zipf.write(file_path, arcname)

    zip_size = output_path.stat().st_size
    print(f"\nArchive created: {output_path}")
    print(f"Size: {zip_size / 1024 / 1024:.1f} MB")

    return output_path


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False  # For speed
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input size


def evaluate_spei_r2_scores(
    gts: np.ndarray,
    preds: np.ndarray
) -> Tuple[float, float, float]:
    """
    Evaluate R² scores for each SPEI target
    
    Args:
        gts: Ground truth values (N x 3)
        preds: Predicted values (N x 3)
    
    Returns:
        Tuple of (SPEI_30d_r2, SPEI_1y_r2, SPEI_2y_r2)
    """
    spei_30d_r2 = r2_score(gts[:, 0], preds[:, 0])
    spei_1y_r2 = r2_score(gts[:, 1], preds[:, 1])
    spei_2y_r2 = r2_score(gts[:, 2], preds[:, 2])
    
    return spei_30d_r2, spei_1y_r2, spei_2y_r2


def evaluate_all_metrics(
    gts: np.ndarray,
    preds: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all metrics for each SPEI target
    
    Args:
        gts: Ground truth values (N x 3)
        preds: Predicted values (N x 3)
    
    Returns:
        Dictionary with all metrics
    """
    targets = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]
    metrics = {}
    
    for i, target in enumerate(targets):
        metrics[target] = {
            "r2": r2_score(gts[:, i], preds[:, i]),
            "mae": mean_absolute_error(gts[:, i], preds[:, i]),
            "rmse": np.sqrt(mean_squared_error(gts[:, i], preds[:, i]))
        }
    
    # Average across all targets
    metrics["average"] = {
        "r2": np.mean([m["r2"] for m in metrics.values() if isinstance(m, dict) and "r2" in m]),
        "mae": np.mean([m["mae"] for m in metrics.values() if isinstance(m, dict) and "mae" in m]),
        "rmse": np.mean([m["rmse"] for m in metrics.values() if isinstance(m, dict) and "rmse" in m])
    }
    
    return metrics


def compile_event_predictions(
    all_gts: np.ndarray,
    all_preds: np.ndarray,
    all_events: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate predictions by event ID (average predictions per event)
    
    Args:
        all_gts: Ground truth values
        all_preds: Predicted values
        all_events: Event IDs
    
    Returns:
        Aggregated ground truths and predictions
    """
    unique_events = np.unique(all_events)
    preds_event = []
    gts_event = []
    
    for event in unique_events:
        indices = np.where(all_events == event)[0]
        if len(indices) == 0:
            continue
        
        preds_event.append(all_preds[indices].mean(axis=0))
        gts_event.append(all_gts[indices].mean(axis=0))
    
    return np.stack(gts_event), np.stack(preds_event)


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: Path,
    is_best: bool = False,
    score: Optional[float] = None
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Validation metrics
        path: Save directory
        is_best: Whether this is the best model
        score: Validation score (R2) for naming
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    
    # Save latest
    torch.save(checkpoint, path / "latest.pth")
    
    # Save best if applicable
    if is_best:
        # cleanup old best checkpoints
        for old_best in path.glob("best_model_r2_*.pth"):
            try:
                old_best.unlink()
            except OSError:
                pass

        if score is not None:
            # Format: best_model_r2_0.1234.pth
            best_name = f"best_model_r2_{score:.4f}.pth"
            weights_name = f"best_weights_r2_{score:.4f}.pth"
        else:
            best_name = "best.pth"
            weights_name = "best_weights.pth"
            
        torch.save(checkpoint, path / best_name)
        # Also copy to generic 'best.pth' for easy resuming
        torch.save(checkpoint, path / "best.pth")
        
        # Cleanup old best weights
        for old_weights in path.glob("best_weights_r2_*.pth"):
            try:
                old_weights.unlink()
            except OSError:
                pass
        
        # Also save just the model weights for inference
        torch.save(model.state_dict(), path / weights_name)
        # Keep generic 'best_weights.pth' for inference scripts that expect it
        torch.save(model.state_dict(), path / "best_weights.pth")


def load_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    weights_only: bool = False
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, Dict]:
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        path: Checkpoint path
        optimizer: Optimizer to load state into
        weights_only: If True, only load weights (for inference)
    
    Returns:
        Tuple of (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(path)
    
    if weights_only:
        # Handle both full checkpoint and weights-only files
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model, None, 0, {}
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model, optimizer, checkpoint["epoch"], checkpoint.get("metrics", {})


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 100,
    warmup_epochs: int = 5
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler
    
    Args:
        optimizer: Optimizer
        scheduler_type: "cosine" or "step"
        epochs: Total epochs
        warmup_epochs: Warmup epochs
    
    Returns:
        LR scheduler
    """
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-7
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    # Wrap with warmup
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs]
        )
    
    return scheduler


def save_results(
    save_path: Path,
    metrics: Dict[str, Dict[str, float]]
):
    """Save evaluation results to JSON"""
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_config(path: Path, args: argparse.Namespace):
    """Save configuration to JSON"""
    config_dict = vars(args)
    # Convert Path objects to strings
    config_dict = {
        k: str(v) if isinstance(v, Path) else v 
        for k, v in config_dict.items()
    }
    
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
