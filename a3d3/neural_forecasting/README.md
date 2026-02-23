# NSF Neural Forecasting Challenge - Team Submission

## Overview

This repository contains our solution for the [NSF Neural Forecasting Challenge](https://www.codabench.org/competitions/4026/) hosted on Codabench. The challenge requires predicting future neural activity (10 timesteps) from historical neural recordings (10 timesteps) across two non-human primate subjects (**Affogato** and **Beignet**), with evaluation on out-of-distribution (OOD) sessions.

**Final Score: Total MSE = 38,134** (improved from baseline 50,257, a reduction of -12,123)

## Approach

Our solution is a **multi-architecture ensemble** with **domain-adaptive training** and **per-domain routing**:

### Architecture Components

| Architecture | Description | Key Features |
|:---|:---|:---|
| **TCNForecasterLarge** | Temporal Convolutional Network | CausalConv1d, BatchNorm, GELU, cross-attention, channel embedding |
| **TransformerForecasterPreLN** | Pre-LayerNorm Transformer | norm_first=True, 8 heads, GELU, learnable positional encoding |
| **NHiTSForecaster** | N-HiTS (hierarchical interpolation) | MaxPool downsampling, multi-scale MLP stacks |

### Per-Domain Strategy

| Domain | Models | Ensemble Weight | Post-Processing |
|:---|:---|:---|:---|
| Affi Public (239ch) | 4 TCN (h=384) + 5 Pre-LN TF (h=384) | TCN 80% / TF 20% | None |
| Affi Private (239ch) | 3 TCN (h=384) | TCN 100% | None |
| Beignet Public (89ch) | 7 TCN (h=256) + 3 Pre-LN TF (h=256) | TCN 60% / TF 40% | None |
| Beignet Private1 (89ch) | 3 TCN (h=256) + NHiTS | 80/20 | margin=0.3, sigma=0.2 |
| Beignet Private2 (89ch) | 3 TCN (h=256) + NHiTS | 80/20 | margin=0.8, sigma=0.5 |

### Key Techniques

1. **CORAL Domain Adaptation**: Aligns covariance and mean statistics between training and target domains to improve OOD generalization
2. **Multi-seed Ensembling**: Training with multiple random seeds and selecting the best-performing subset
3. **Domain Detection**: Automatic classification of test sessions into Public/Private domains using channel statistics (MAD threshold for Affi, statistical signatures for Beignet 3-way)
4. **Per-domain Post-processing**: Gaussian smoothing with domain-specific margin/sigma parameters (only for Beignet Private)
5. **Feature Engineering**: Up to 9 features including raw signal, temporal differences, rolling statistics, and z-scores

## Score Progression

| Version | Total MSE | Key Strategy |
|:---:|:---:|:---|
| Baseline | 50,257 | Transformer d=192 |
| v80 | 47,625 | TCN ensemble d=384 |
| v132 | 45,642 | 9-feature engineering |
| v176 | 42,617 | Domain routing |
| v192b | 41,967 | CORAL domain adaptation |
| v204 | 39,051 | Multi-seed + PP sweep |
| v221 | 38,252 | Pre-LN Transformer 3-seed |
| v226c | 38,214 | Affi Pre-LN TF 80/20 |
| v232c | 38,157 | Private h=256 + CORAL 3.5/1.5 |
| **v233b** | **38,134** | **TCN/TF 60/40 Beignet Public** |

## Repository Structure

```
.
├── baselines/
│   ├── submissions/
│   │   └── v233b_ensemble/          # Final submission (inference code)
│   │       ├── model.py             # Inference entry point
│   │       ├── requirements.txt
│   │       └── README.md
│   └── training/                    # Training code by model type
│       ├── affi_tcn/
│       ├── affi_transformer/
│       ├── beignet_tcn/
│       ├── beignet_transformer/
│       ├── beignet_private/
│       └── nhits/
├── model_weights/                   # Trained model weights (Git LFS)
│   ├── affi/                        # 9 models + 1 normalization
│   ├── beignet_public/              # 11 models + 2 normalizations
│   └── beignet_private/             # 6 models + 1 normalization
├── notebooks/                       # Training notebooks (Google Colab)
├── .gitattributes                   # Git LFS tracking
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0

## How to Run Inference

The submission is designed to run on the Codabench evaluation platform. The entry point is `baselines/submissions/v233b_ensemble/model.py`, which:

1. Loads all model weights from the submission directory
2. Detects the subject (Affogato or Beignet) based on channel count
3. Detects the domain (Public/Private) using statistical thresholds
4. Routes to the appropriate ensemble and post-processing pipeline
5. Outputs predictions of shape `(10, n_channels)` for each test sample

```python
# The model.py follows the competition's API:
# - Receives test data as numpy arrays
# - Returns predictions as numpy arrays
```

## Training

Training was performed on Google Colab (A100 GPU). Each model type has its own training script in `baselines/training/` and corresponding Colab notebook in `notebooks/`.

Key training details:
- **Optimizer**: AdamW with cosine annealing
- **Loss**: MSE + CORAL alignment loss (lambda varies by model)
- **Data split**: Walk-forward validation on later sessions
- **Augmentation**: Gaussian noise injection, temporal jittering

## License

MIT License
