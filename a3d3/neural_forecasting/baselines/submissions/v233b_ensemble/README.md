# v233b Ensemble Submission

**Total MSE: 38,134** on Codabench evaluation

## Model Composition (26 models)

### Affogato (239 channels)
- 4x TCNForecasterLarge (h=384): CORAL and Original domain variants, 1-feat and 3-feat
- 5x TransformerForecasterPreLN (h=384, L=4, 8 heads): CORAL-trained, seeds [42, 123, 456, 789, 2024]

### Beignet Public (89 channels)
- 7x TCNForecasterLarge (h=256, L=3): Multi-seed CORAL, 9-feat, dropout=0.35
- 3x TransformerForecasterPreLN (h=256, L=4, 8 heads): Pre-LN, seeds [789, 42, 2024]
- 1x NHiTSForecaster (h=256, L=2): CORAL-trained, 1-feat

### Beignet Private (89 channels)
- 6x TCNForecaster (h=256, L=3): 3 seeds per private domain, dropout=0.4

## Inference

The `model.py` file is the entry point. It:
1. Auto-detects subject by channel count (239=Affi, 89=Beignet)
2. Auto-detects domain (Public/Private) using statistical thresholds
3. Applies per-domain ensemble weights and post-processing
