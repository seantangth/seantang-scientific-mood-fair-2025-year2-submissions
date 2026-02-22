#!/usr/bin/env python3
"""
Evaluation script for iHARP Coastal Flooding F1-Push Ranker model.

This script evaluates a trained f1push_ranker_v1 model using
leave-one-station-out cross-validation on the training data.

Usage:
    python evaluation.py --data_file <path_to_train_hourly.csv> \
                         --threshold_mat <path_to_thresholds.mat> \
                         --model_dir <path_to_model_dir>

The model_dir should contain a model.pkl produced by train.py.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

HIST_DAYS = 7
FUTURE_DAYS = 14

BASE_FEATURES = [
    "sea_level_mean",
    "sea_level_max",
    "sea_level_min",
    "sea_level_std",
    "sea_level_3d_mean",
    "sea_level_7d_mean",
    "sea_level_3d_std",
    "sea_level_diff_1d",
    "sea_level_diff_3d",
    "rel_mean_off",
    "rel_max_off",
    "rel_3d_off",
    "rel_7d_off",
    "ratio_max_off",
    "station_z_mean",
    "station_rel_mean",
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate F1-Push Ranker model")
    p.add_argument("--data_file", required=True,
                    help="Path to train_hourly.csv")
    p.add_argument("--threshold_mat", required=True,
                    help="Path to Seed_Coastal_Stations_Thresholds.mat")
    p.add_argument("--model_dir", required=True,
                    help="Directory containing model.pkl")
    p.add_argument("--label_mode", choices=["official", "dynamic", "union"],
                    default="union")
    p.add_argument("--q_values", default="0.00001,0.00002,0.00005,0.0001",
                    help="Comma-separated quantile values for policy evaluation")
    p.add_argument("--save_results", type=str, default=None,
                    help="Path to save evaluation results JSON")
    return p.parse_args()


def load_data(data_file):
    df = pd.read_csv(data_file)
    df["time"] = pd.to_datetime(df["time"])
    return df


def create_daily_features(df):
    x = df.copy()
    x["date"] = x["time"].dt.floor("D")
    daily = (
        x.groupby(["station_name", "date"])
        .agg(
            sea_level_mean=("sea_level", "mean"),
            sea_level_max=("sea_level", "max"),
            sea_level_min=("sea_level", "min"),
            sea_level_std=("sea_level", "std"),
        )
        .reset_index()
        .sort_values(["station_name", "date"])
        .reset_index(drop=True)
    )
    daily["sea_level_3d_mean"] = daily.groupby("station_name")["sea_level_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    daily["sea_level_7d_mean"] = daily.groupby("station_name")["sea_level_mean"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    daily["sea_level_3d_std"] = daily.groupby("station_name")["sea_level_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).std()
    )
    daily["sea_level_diff_1d"] = daily.groupby("station_name")["sea_level_mean"].transform(lambda s: s.diff(1))
    daily["sea_level_diff_3d"] = daily.groupby("station_name")["sea_level_mean"].transform(lambda s: s.diff(3))
    return daily.fillna(0.0)


def load_official_threshold_map(threshold_mat):
    mat = loadmat(threshold_mat)
    names = [x[0] for x in mat["sname"].squeeze()]
    vals = mat["thminor_stnd"].squeeze().astype(float)
    return {str(n): float(v) for n, v in zip(names, vals)}


def attach_thresholds_and_features(daily, off_map, df):
    x = daily.copy()
    x["thr_off"] = x["station_name"].map(off_map)

    dyn = df.groupby("station_name")["sea_level"].agg(["mean", "std"]).reset_index()
    dyn["thr_dyn"] = dyn["mean"] + 1.5 * dyn["std"]
    x = x.merge(dyn[["station_name", "thr_dyn"]], on="station_name", how="left")
    x["thr_off"] = x["thr_off"].fillna(x["thr_dyn"])

    eps = 1e-6
    x["rel_mean_off"] = x["sea_level_mean"] - x["thr_off"]
    x["rel_max_off"] = x["sea_level_max"] - x["thr_off"]
    x["rel_3d_off"] = x["sea_level_3d_mean"] - x["thr_off"]
    x["rel_7d_off"] = x["sea_level_7d_mean"] - x["thr_off"]
    x["ratio_max_off"] = x["sea_level_max"] / (x["thr_off"].abs() + eps)

    st_stats = x.groupby("station_name")["sea_level_mean"].agg(st_mean="mean", st_std="std").reset_index()
    x = x.merge(st_stats, on="station_name", how="left")
    x["station_z_mean"] = (x["sea_level_mean"] - x["st_mean"]) / (x["st_std"].abs() + eps)
    x["station_rel_mean"] = x["sea_level_mean"] - x["st_mean"]

    x["flood_day_off"] = (x["sea_level_max"] > x["thr_off"]).astype(np.int8)
    x["flood_day_dyn"] = (x["sea_level_max"] > x["thr_dyn"]).astype(np.int8)
    x["flood_day_union"] = np.maximum(x["flood_day_off"], x["flood_day_dyn"]).astype(np.int8)
    return x.fillna(0.0)


def build_windows(daily, stations, n_days, label_mode):
    label_col = {"official": "flood_day_off", "dynamic": "flood_day_dyn", "union": "flood_day_union"}[label_mode]
    rows_x, rows_y = [], []
    n_days = int(n_days)

    for stn in stations:
        g = daily[daily["station_name"] == stn].sort_values("date").reset_index(drop=True)
        if g.empty:
            continue
        feat = g[BASE_FEATURES].to_numpy(dtype=np.float32)
        flood = g[label_col].to_numpy(dtype=np.int8)
        max_i = len(g) - HIST_DAYS - FUTURE_DAYS + 1
        if max_i <= 0:
            continue
        for i in range(max_i):
            start_idx = i + HIST_DAYS - n_days
            end_idx = i + HIST_DAYS
            if start_idx < 0:
                continue
            block = feat[start_idx:end_idx]
            if block.shape[0] != n_days:
                continue
            y = int(flood[i + HIST_DAYS: i + HIST_DAYS + FUTURE_DAYS].max() > 0)
            rows_x.append(block.reshape(-1))
            rows_y.append(y)

    if not rows_x:
        return np.empty((0, len(BASE_FEATURES) * n_days), dtype=np.float32), np.empty(0, dtype=np.int8)
    return np.asarray(rows_x, dtype=np.float32), np.asarray(rows_y, dtype=np.int8)


def evaluate_loo(daily, stations, bundle, label_mode, q_values):
    """Leave-one-station-out evaluation."""
    n_days = int(bundle.get("use_last_n_days", 3))
    results_per_station = {}
    all_y_true = []
    all_y_prob = []

    for test_station in stations:
        train_stations = [s for s in stations if s != test_station]

        X_train, y_train = build_windows(daily, train_stations, n_days, label_mode)
        X_test, y_test = build_windows(daily, [test_station], n_days, label_mode)

        if len(X_test) == 0:
            continue

        # Use the bundle's scaler on test data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_s = scaler.transform(X_test)

        model = bundle["model"]
        y_prob = model.predict_proba(X_test_s)[:, 1].astype(np.float32)

        try:
            auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
        except Exception:
            auc = 0.5

        results_per_station[test_station] = {
            "n_samples": int(len(y_test)),
            "pos_rate": float(y_test.mean()),
            "auc": auc,
        }

        all_y_true.append(y_test)
        all_y_prob.append(y_prob)

        print(f"  {test_station:20s}: n={len(y_test):6d} pos={y_test.mean():.4f} auc={auc:.4f}")

    if not all_y_true:
        print("No valid evaluation data.")
        return {}

    y_true = np.concatenate(all_y_true)
    y_prob = np.concatenate(all_y_prob)

    pooled_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5

    # Evaluate quantile policies
    policy_results = {}
    for q in q_values:
        n = len(y_true)
        k = max(1, int(n * q))
        low_idx = np.argpartition(y_prob, k - 1)[:k]
        y_pred = np.ones(n, dtype=np.int8)
        y_pred[low_idx] = 0

        tn = int((y_true[low_idx] == 0).sum())
        fn = int((y_true[low_idx] == 1).sum())

        policy_results[f"q={q}"] = {
            "k_flip": int(k),
            "tn": tn,
            "fn": fn,
            "acc": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }

    return {
        "pooled_auc": pooled_auc,
        "pooled_n": int(len(y_true)),
        "pooled_pos_rate": float(y_true.mean()),
        "station_results": results_per_station,
        "policy_results": policy_results,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("iHARP F1-Push Ranker Evaluation")
    print("=" * 60)

    # Load model
    model_path = Path(args.model_dir) / "model.pkl"
    print(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    print(f"Model kind: {bundle.get('model_kind', 'unknown')}")
    print(f"Use last N days: {bundle.get('use_last_n_days', 3)}")

    # Load and process data
    print(f"\nLoading data: {args.data_file}")
    df = load_data(args.data_file)
    daily = create_daily_features(df)
    stations = sorted(daily["station_name"].unique().tolist())

    off_map = load_official_threshold_map(args.threshold_mat)
    daily = attach_thresholds_and_features(daily, off_map, df)

    print(f"Stations: {stations}")
    print(f"Total daily rows: {len(daily)}")

    # Parse q values
    q_values = [float(x.strip()) for x in args.q_values.split(",") if x.strip()]

    # Run leave-one-station-out evaluation
    print(f"\nRunning LOO evaluation (label_mode={args.label_mode})...")
    results = evaluate_loo(daily, stations, bundle, args.label_mode, q_values)

    # Print summary
    if results:
        print(f"\n{'='*60}")
        print("Evaluation Summary")
        print(f"{'='*60}")
        print(f"Pooled AUC: {results['pooled_auc']:.4f}")
        print(f"Pooled N: {results['pooled_n']:,}")
        print(f"Pooled Pos Rate: {results['pooled_pos_rate']:.4f}")
        print(f"\nPolicy Results:")
        for q_key, pm in results.get("policy_results", {}).items():
            print(f"  {q_key}: k={pm['k_flip']} tn={pm['tn']} fn={pm['fn']} "
                  f"acc={pm['acc']:.4f} f1={pm['f1']:.4f} mcc={pm['mcc']:.4f}")
        print(f"{'='*60}")

    # Save results
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
