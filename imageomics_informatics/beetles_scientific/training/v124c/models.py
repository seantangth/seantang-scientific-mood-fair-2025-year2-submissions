"""
Model architectures for HDR-SMood Beetles Challenge
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from open_clip import create_model_and_transforms
from typing import List, Optional


class RegressorHead(nn.Module):
    """
    Multi-layer regressor head for SPEI prediction
    """
    def __init__(
        self,
        in_features: int = 768,
        hidden_sizes: List[int] = [512, 128, 32],
        num_outputs: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_size = in_features
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Final output layer (no dropout after)
        layers.append(nn.Linear(prev_size, num_outputs))
        
        self.regressor = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class BioCLIP2Regressor(nn.Module):
    """
    BioCLIP v2 backbone with regression head
    """
    def __init__(
        self,
        hidden_sizes: List[int] = [512, 128, 32],
        num_outputs: int = 3,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load BioCLIP
        self.bioclip, _, self.preprocess = create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2",
            output_dict=True,
            require_pretrained=True
        )
        self.bioclip = self.bioclip.to(device)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.bioclip.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        num_features = 768  # BioCLIP output dimension
        
        # Regression head
        self.regressor = RegressorHead(
            in_features=num_features,
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout
        )
        
        self.freeze_backbone = freeze_backbone
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frozen backbone"""
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.bioclip(x)["image_features"]
        else:
            features = self.bioclip(x)["image_features"]
        
        return self.regressor(features)
    

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features only (for pre-extraction)"""
        with torch.no_grad():
            return self.bioclip(x)["image_features"]


class BioCLIP2MetadataRegressor(nn.Module):
    """
    BioCLIP v2 backbone + Metadata (Month) fusion
    """
    def __init__(
        self,
        hidden_sizes: List[int] = [512, 128, 32],
        num_outputs: int = 3,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load BioCLIP
        self.bioclip, _, self.preprocess = create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2",
            output_dict=True,
            require_pretrained=True
        )
        self.bioclip = self.bioclip.to(device)
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.bioclip.parameters():
                param.requires_grad = False
        
        num_image_features = 768
        
        # Metadata encoder (Month sin/cos -> 64d)
        self.meta_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined regressor
        combined_features = num_image_features + 64
        self.regressor = RegressorHead(
            in_features=combined_features,
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout
        )
        
        self.freeze_backbone = freeze_backbone
    
    def forward(self, x: torch.Tensor, month_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with image and metadata
        x: Image tensor [B, C, H, W]
        month_feats: Month sin/cos [B, 2]
        """
        if self.freeze_backbone:
            with torch.no_grad():
                img_features = self.bioclip(x)["image_features"]
        else:
            img_features = self.bioclip(x)["image_features"]
            
        # Encode metadata
        meta_features = self.meta_encoder(month_feats)
        
        # Concatenate
        combined = torch.cat([img_features, meta_features], dim=1)
        
        return self.regressor(combined)



class DINOv2Regressor(nn.Module):
    """
    DINOv2 backbone with regression head
    Uses convolutional layers to process patch tokens
    """
    def __init__(
        self,
        hidden_sizes: List[int] = [512, 128, 32],
        num_outputs: int = 3,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load DINOv2
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base")
        self.dino = self.dino.to(device)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False
        
        # Convolutional layers to process patch tokens
        # Input: B x 768 x 16 x 16 (reshaped from B x 256 x 768)
        self.tokens_to_linear = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=5, stride=1, padding=0),  # -> B x 768 x 12 x 12
            nn.ReLU(),
            nn.Conv2d(768, 1024, kernel_size=12, stride=1, padding=0),  # -> B x 1024 x 1 x 1
            nn.ReLU(),
        )
        
        # Regression head
        self.regressor = RegressorHead(
            in_features=1024,
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout
        )
        
        self.freeze_backbone = freeze_backbone
    
    def _reshape_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape DINOv2 patch tokens from B x 256 x 768 to B x 768 x 16 x 16"""
        transposed = features.transpose(1, 2).contiguous()  # B x 768 x 256
        unflat = transposed.unflatten(dim=2, sizes=(16, 16))  # B x 768 x 16 x 16
        return unflat
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.dino(x)[0][:, 1:]  # Remove CLS token
        else:
            features = self.dino(x)[0][:, 1:]
        
        # Reshape and process
        reshaped = self._reshape_tokens(features)
        pooled = self.tokens_to_linear(reshaped).squeeze(-1).squeeze(-1)
        
        return self.regressor(pooled)
    
    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Forward from pre-extracted features (B x 768 x 16 x 16)"""
        pooled = self.tokens_to_linear(features).squeeze(-1).squeeze(-1)
        return self.regressor(pooled)


class EnsembleRegressor(nn.Module):
    """
    Ensemble of multiple models
    """
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average of model predictions"""
        outputs = []
        for model, weight in zip(self.models, self.weights):
            outputs.append(model(x) * weight)
        
        return torch.stack(outputs).sum(dim=0)


class BioCLIPSpeciesAwareRegressor(nn.Module):
    """
    BioCLIP v2 + Species/Domain Embeddings
    Takes species_id and domain_id as additional inputs
    """
    def __init__(
        self,
        num_species: int = 200,
        num_domains: int = 20,
        species_embed_dim: int = 64,
        domain_embed_dim: int = 32,
        hidden_sizes: List[int] = [512, 128, 32],
        num_outputs: int = 3,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load BioCLIP
        self.bioclip, _, self.preprocess = create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2",
            output_dict=True,
            require_pretrained=True
        )
        self.bioclip = self.bioclip.to(device)
        
        if freeze_backbone:
            for param in self.bioclip.parameters():
                param.requires_grad = False
        
        num_image_features = 768
        
        # Species embedding
        self.species_embedding = nn.Embedding(num_species, species_embed_dim)
        
        # Domain embedding
        self.domain_embedding = nn.Embedding(num_domains, domain_embed_dim)
        
        # Combined regressor (image + species + domain)
        combined_features = num_image_features + species_embed_dim + domain_embed_dim
        self.regressor = RegressorHead(
            in_features=combined_features,
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout
        )
        
        self.freeze_backbone = freeze_backbone
    
    def forward(
        self, 
        x: torch.Tensor, 
        species_ids: torch.Tensor, 
        domain_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with image and metadata embeddings
        x: Image tensor [B, C, H, W]
        species_ids: Species indices [B]
        domain_ids: Domain indices [B]
        """
        if self.freeze_backbone:
            with torch.no_grad():
                img_features = self.bioclip(x)["image_features"]
        else:
            img_features = self.bioclip(x)["image_features"]
        
        # Get embeddings
        species_emb = self.species_embedding(species_ids)  # [B, species_embed_dim]
        domain_emb = self.domain_embedding(domain_ids)    # [B, domain_embed_dim]
        
        # Concatenate all features
        combined = torch.cat([img_features, species_emb, domain_emb], dim=1)
        
        return self.regressor(combined)


class DINOv2SpeciesAwareRegressor(nn.Module):
    """
    DINOv2 + Species/Domain Embeddings
    """
    def __init__(
        self,
        num_species: int = 200,
        num_domains: int = 20,
        species_embed_dim: int = 64,
        domain_embed_dim: int = 32,
        hidden_sizes: List[int] = [512, 128, 32],
        num_outputs: int = 3,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load DINOv2
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base")
        self.dino = self.dino.to(device)
        
        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False
        
        # Conv layers for patch tokens
        self.tokens_to_linear = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(768, 512, kernel_size=12, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Species and domain embeddings
        self.species_embedding = nn.Embedding(num_species, species_embed_dim)
        self.domain_embedding = nn.Embedding(num_domains, domain_embed_dim)
        
        # Combined regressor
        combined_features = 512 + species_embed_dim + domain_embed_dim
        self.regressor = RegressorHead(
            in_features=combined_features,
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout
        )
        
        self.freeze_backbone = freeze_backbone
    
    def _reshape_tokens(self, features: torch.Tensor) -> torch.Tensor:
        transposed = features.transpose(1, 2).contiguous()
        unflat = transposed.unflatten(dim=2, sizes=(16, 16))
        return unflat
    
    def forward(
        self, 
        x: torch.Tensor, 
        species_ids: torch.Tensor, 
        domain_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.dino(x)[0][:, 1:]
        else:
            features = self.dino(x)[0][:, 1:]
        
        reshaped = self._reshape_tokens(features)
        img_features = self.tokens_to_linear(reshaped).squeeze(-1).squeeze(-1)
        
        species_emb = self.species_embedding(species_ids)
        domain_emb = self.domain_embedding(domain_ids)
        
        combined = torch.cat([img_features, species_emb, domain_emb], dim=1)
        
        return self.regressor(combined)


def get_model(
    backbone: str = "bioclip",
    hidden_sizes: List[int] = [512, 128, 32],
    num_outputs: int = 3,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
    device: str = "cuda",
    # New parameters for species-aware models
    num_species: int = 200,
    num_domains: int = 20,
    species_embed_dim: int = 64,
    domain_embed_dim: int = 32,
) -> nn.Module:
    """
    Factory function to create model
    
    Args:
        backbone: "bioclip", "dinov2", "bioclip_meta", "bioclip_species", "dinov2_species"
        hidden_sizes: Hidden layer sizes for regressor
        num_outputs: Number of output values
        dropout: Dropout rate
        freeze_backbone: Whether to freeze backbone
        device: Device to use
        num_species: Number of unique species (for species-aware models)
        num_domains: Number of unique domains (for species-aware models)
        species_embed_dim: Species embedding dimension
        domain_embed_dim: Domain embedding dimension
    
    Returns:
        Model instance
    """
    if backbone == "bioclip":
        model = BioCLIP2Regressor(
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            device=device
        )
    elif backbone == "dinov2":
        model = DINOv2Regressor(
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            device=device
        )
    elif backbone == "bioclip_meta":
        model = BioCLIP2MetadataRegressor(
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            device=device
        )
    elif backbone == "bioclip_species":
        model = BioCLIPSpeciesAwareRegressor(
            num_species=num_species,
            num_domains=num_domains,
            species_embed_dim=species_embed_dim,
            domain_embed_dim=domain_embed_dim,
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            device=device
        )
    elif backbone == "dinov2_species":
        model = DINOv2SpeciesAwareRegressor(
            num_species=num_species,
            num_domains=num_domains,
            species_embed_dim=species_embed_dim,
            domain_embed_dim=domain_embed_dim,
            hidden_sizes=hidden_sizes,
            num_outputs=num_outputs,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            device=device
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    return model.to(device)

