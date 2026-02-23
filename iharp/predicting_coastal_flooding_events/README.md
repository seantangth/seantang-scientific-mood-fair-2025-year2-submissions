# iHARP Coastal Flooding - Final Submission Repo (hybrid_final_v1)

This repository is prepared to follow the official sample-style layout:
- `submission/` contains the Codabench submission files (`model.py`, `requirements.txt`, model artifacts).

## Repository Structure

- `submission/`
  - `model.py` — Hybrid-Final: d2s2 + G7v2 ensemble (per-window rank blend, per-day G7v2)
  - `requirements.txt`
  - `bundle_00.pkl` ... `bundle_04.pkl` — F1-Push Ranker bundles
  - `booster.json` — XGBoost G7v2 booster weights
  - `scaler_stats.npz` — Feature normalization statistics
  - `inference_meta.json` — Inference metadata
- `final_submission/hybrid_final_v1.zip`
  - exact zip package used for final Codabench submission
- `training/`
  - `train.py` — Full F1-Push Ranker training script
  - `evaluation.py` — LOO cross-validation evaluation

## Final Package

- Final submission package: `final_submission/hybrid_final_v1.zip`
- Strategy: Hybrid ensemble — rank-based blend of d2s2 F1-Push Ranker (0.6) and G7v2 XGBoost (0.4) for per-window mode; G7v2-only for per-day mode

## Local Inference Example

Run from repo root:

```bash
python3 submission/model.py \
  --train_hourly <path_to_train_hourly.csv> \
  --test_hourly <path_to_test_hourly.csv> \
  --test_index <path_to_test_index.csv> \
  --predictions_out predictions.csv
```

Expected output:
- CSV with columns: `id`, `y_prob`
