#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_CSV="$ROOT_DIR/tmp_d2s2_predictions.csv"

# Default paths target the original project layout.
TRAIN_HOURLY="${TRAIN_HOURLY:-$ROOT_DIR/../../../1_data/processed/train_hourly.csv}"
TEST_HOURLY="${TEST_HOURLY:-$ROOT_DIR/../../../1_data/processed/test_hourly.csv}"
TEST_INDEX="${TEST_INDEX:-$ROOT_DIR/../../../tmp_rovodev_smoke/ingestion_work/test_index.csv}"

python3 "$ROOT_DIR/submission/model.py" \
  --train_hourly "$TRAIN_HOURLY" \
  --test_hourly "$TEST_HOURLY" \
  --test_index "$TEST_INDEX" \
  --predictions_out "$OUT_CSV"

echo "Smoke test done: $OUT_CSV"
head -n 5 "$OUT_CSV"
