# iHARP Coastal Flooding - Final Submission Repo (d2s2_q00001_v1)

This repository is prepared to follow the official sample-style layout:
- `submission/` contains the Codabench submission files (`model.py`, `requirements.txt`, model artifacts).

## Repository Structure

- `submission/`
  - `model.py`
  - `requirements.txt`
  - `bundle_00.pkl` ... `bundle_04.pkl`
  - `model_meta.json`
  - `README_submission.md`
- `final_submission/d2s2_q00001_v1.zip`
  - exact zip package used for final Codabench submission
- `scripts/create_station_quota_consensus_submissions.py`
  - original D2 packer script used to create `d2s2_*` packages
- `scripts/smoke_test_d2s2.sh`
  - local smoke test helper
- `docs/`
  - project tracking docs containing D2 route and online evaluation records

## Final Package

- Final submission package: `final_submission/d2s2_q00001_v1.zip`
- Strategy: station-quota consensus low-risk selection
- `submission_q`: `1e-05`
- Sources:
  - `f1push_ranker_v1_20260214_075444`
  - `a2_balanced_xgb_v1_20260214_113741`
  - `b1_groupdro_xgb_v1_20260214_122519`
  - `b2_union_xgb_v1_20260214_130826`
  - `a3_focal_xgb_v1_20260215_122316`

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
