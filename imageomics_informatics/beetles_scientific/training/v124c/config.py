"""
Configuration for HDR-SMood Beetles Challenge Training
"""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    backbone: str = "bioclip"  # "bioclip" or "dinov2"
    num_features: int = 768  # Feature dimension from backbone
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 128, 32])
    num_outputs: int = 3  # SPEI_30d, SPEI_1y, SPEI_2y
    dropout: float = 0.1
    freeze_backbone: bool = True
    n_trainable_blocks: int = 0  # Number of backbone blocks to fine-tune (0 = all frozen)


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 5
    early_stopping_patience: int = 20
    gradient_clip: float = 1.0
    num_workers: int = 4
    mixed_precision: bool = True  # Use AMP on H100


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "imageomics/sentinel-beetles"
    hf_token: Optional[str] = None
    image_size: int = 224
    # Normalization (ImageNet defaults for CLIP/DINOv2)
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)


@dataclass
class PathConfig:
    """Path configuration"""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "1_data")
    model_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "4_models")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "5_outputs")
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "5_outputs" / "logs")


@dataclass
class Config:
    """Main configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Experiment settings
    experiment_name: str = "baseline"
    seed: int = 42
    device: str = "cuda"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.paths.model_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)


def get_config(**kwargs) -> Config:
    """Get configuration with optional overrides"""
    config = Config()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)
    
    return config
