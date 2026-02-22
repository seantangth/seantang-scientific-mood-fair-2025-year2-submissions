"""
Data loading utilities for HDR-SMood Beetles Challenge
"""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoImageProcessor
from open_clip import create_model_and_transforms
from typing import Tuple, Optional, Callable, Dict
import numpy as np


# ============================================================
# SPECIES AND DOMAIN ID MAPPINGS
# ============================================================

# Domain ID mapping (10 domains in training data)
DOMAIN_TO_ID = {
    1: 0, 3: 1, 4: 2, 7: 3, 9: 4, 
    11: 5, 32: 6, 46: 7, 99: 8, 202: 9
}
NUM_DOMAINS = len(DOMAIN_TO_ID) + 1  # +1 for unknown

# Species name to ID mapping (will be built from data)
# This is initialized with common species and updated during data loading
SPECIES_TO_ID: Dict[str, int] = {}
NUM_SPECIES = 200  # Reserve space for up to 200 species


def build_species_mapping(dataset) -> Dict[str, int]:
    """Build species name to ID mapping from dataset"""
    global SPECIES_TO_ID
    
    species_set = set()
    for split in dataset.keys():
        for example in dataset[split]:
            species_set.add(example["scientificName"])
    
    SPECIES_TO_ID = {name: idx for idx, name in enumerate(sorted(species_set))}
    return SPECIES_TO_ID


def get_species_id(name: str) -> int:
    """Get species ID from name, return 0 (unknown) if not found"""
    return SPECIES_TO_ID.get(name, 0)


def get_domain_id(domain: int) -> int:
    """Get domain ID from raw domain value, return 0 (unknown) if not found"""
    return DOMAIN_TO_ID.get(domain, 0)


def get_transforms(backbone: str = "bioclip"):
    """
    Get image transforms based on backbone type
    
    Args:
        backbone: "bioclip" or "dinov2"
    
    Returns:
        transform function and processor (if needed)
    """
    if backbone == "bioclip" or backbone == "bioclip_meta":
        _, _, preprocess = create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2",
            output_dict=True,
            require_pretrained=True
        )
        return preprocess, None
    
    elif backbone == "dinov2":
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        
        def transform(img):
            return processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        
        return transform, processor
    
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def get_collate_fn(
    other_columns: Optional[list] = None,
    include_month_features: bool = False,
    include_species_domain: bool = False,
) -> Callable:
    """Create collate function for DataLoader.

    The HuggingFace dataset rows include non-tensor objects like PIL Images
    (e.g. `file_path`). PyTorch's default collator can't batch those.

    This collate function explicitly picks the tensor/numeric columns we need.

    Args:
        other_columns: Additional columns to include in batch (as 1D tensors)
        include_month_features: If True, also return month_sin/cos as 3rd element
        include_species_domain: If True, return species_id and domain_id tensors

    Returns:
        Collate function returning a list: 
        - Default: [pixel_values, spei_values]
        - With species_domain: [pixel_values, spei_values, species_ids, domain_ids]
    """

    def collate_fn(batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        spei_values = torch.stack(
            [
                torch.tensor(
                    [example["SPEI_30d"], example["SPEI_1y"], example["SPEI_2y"]],
                    dtype=torch.float32,
                )
                for example in batch
            ]
        )

        result = [pixel_values, spei_values]

        if include_species_domain:
            # Get species and domain IDs
            species_ids = torch.tensor(
                [example.get("species_id", 0) for example in batch],
                dtype=torch.long
            )
            domain_ids = torch.tensor(
                [example.get("domain_id", 0) for example in batch],
                dtype=torch.long
            )
            result.extend([species_ids, domain_ids])

        if include_month_features:
            month_values = torch.stack(
                [
                    torch.tensor(
                        [example.get("month_sin", 0.0), example.get("month_cos", 0.0)],
                        dtype=torch.float32,
                    )
                    for example in batch
                ]
            )
            result.append(month_values)

        if other_columns:
            for col in other_columns:
                result.append(torch.tensor([example[col] for example in batch]))

        return result

    return collate_fn


def load_beetle_dataset(
    hf_token: Optional[str] = None,
    backbone: str = "bioclip",
    build_species_map: bool = True
) -> Tuple:
    """
    Load Sentinel Beetles dataset from Hugging Face
    
    Args:
        hf_token: Hugging Face token (if needed)
        backbone: Backbone type for transforms
        build_species_map: If True, build species name to ID mapping
    
    Returns:
        train_dataset, val_dataset, transform
    """
    global SPECIES_TO_ID
    
    # Load dataset
    ds = load_dataset(
        "imageomics/sentinel-beetles",
        token=hf_token
    )
    
    # Build species mapping if requested
    if build_species_map:
        print("Building species mapping from dataset...")
        species_set = set()
        for example in ds["train"]:
            species_set.add(example["scientificName"])
        for example in ds["validation"]:
            species_set.add(example["scientificName"])
        
        # Sort for reproducibility, reserve 0 for unknown
        SPECIES_TO_ID = {name: idx + 1 for idx, name in enumerate(sorted(species_set))}
        print(f"Found {len(SPECIES_TO_ID)} unique species")
    
    # Get transforms
    transform, _ = get_transforms(backbone)
    
    # Pre-calculate month features
    def parse_month_features(date_str):
        if not date_str:
            return 0.0, 0.0
        try:
            month = int(date_str.split('-')[1])
            angle = 2 * np.pi * (month - 1) / 12
            return np.sin(angle), np.cos(angle)
        except:
            return 0.0, 0.0

    # Apply transforms with species/domain IDs
    def dset_transforms(examples):
        pixel_values = [
            transform(img.convert("RGB")) for img in examples["file_path"]
        ]
        examples["pixel_values"] = pixel_values
        
        # Add month features
        month_feats = [parse_month_features(d) for d in examples["collectDate"]]
        examples["month_sin"] = [f[0] for f in month_feats]
        examples["month_cos"] = [f[1] for f in month_feats]
        
        # Add species ID (0 if unknown)
        examples["species_id"] = [
            SPECIES_TO_ID.get(name, 0) for name in examples["scientificName"]
        ]
        
        # Add domain ID (using DOMAIN_TO_ID mapping)
        examples["domain_id"] = [
            DOMAIN_TO_ID.get(did, 0) for did in examples["domainID"]
        ]
        
        return examples
    
    train_dset = ds["train"].with_transform(dset_transforms)
    val_dset = ds["validation"].with_transform(dset_transforms)
    
    return train_dset, val_dset, transform


def create_dataloaders(
    train_dset,
    val_dset,
    batch_size: int = 64,
    num_workers: int = 4,
    other_columns: Optional[list] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation
    
    Args:
        train_dset: Training dataset
        val_dset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers
        other_columns: Additional columns to include
    
    Returns:
        train_dataloader, val_dataloader
    """
    # Updated collate function to include month features by default
    def collate_fn(batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        spei_values = torch.stack([
            torch.tensor([
                example["SPEI_30d"],
                example["SPEI_1y"],
                example["SPEI_2y"]
            ], dtype=torch.float32)
            for example in batch
        ])
        
        # Add month features: [B, 2]
        month_values = torch.stack([
            torch.tensor([
                example["month_sin"],
                example["month_cos"]
            ], dtype=torch.float32)
            for example in batch
        ])
        
        result = [pixel_values, spei_values, month_values]
        
        if other_columns:
            for col in other_columns:
                result.append(torch.tensor([example[col] for example in batch]))
        
        return result
    
    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def extract_features(
    dataloader: DataLoader,
    model,
    backbone: str = "bioclip",
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from backbone model
    
    Args:
        dataloader: DataLoader with images
        model: Backbone model (BioCLIP or DINOv2)
        backbone: Backbone type
        device: Device to use
    
    Returns:
        Features tensor, Labels tensor
    """
    from tqdm import tqdm
    
    X = []
    Y = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            x, y = batch[:2]
            x = x.to(device)
            
            if backbone == "bioclip":
                features = model(x)["image_features"]
            elif backbone == "dinov2":
                # DINOv2 outputs patch tokens, reshape them
                features = model(x)[0][:, 1:]  # Remove CLS token
                # Reshape Bx256x768 -> Bx768x16x16
                features = features.transpose(1, 2)
                features = features.unflatten(dim=2, sizes=(16, 16))
            
            X.append(features.cpu())
            Y.append(y)
    
    X = torch.cat(X)
    Y = torch.cat(Y)
    
    return X, Y
