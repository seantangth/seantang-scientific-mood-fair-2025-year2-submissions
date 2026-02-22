#!/usr/bin/env python3
"""
F1-Push Ranker Training Suite v1
================================
Goal:
- Build a new model line specialized for F1-first leaderboard behavior.
- Keep "predict flood" as default and learn a robust ranking for rare
  non-flood candidates (for tiny quantile flip policies).

Outputs:
- 4_models/f1push_ranker_v1_<timestamp>/
  - model.pkl
  - model_meta.json
  - cv_policy_results.csv
  - results.json
  - run_note.md
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise SystemExit(f"XGBoost is required for this script: {exc}")


RANDOM_SEED = 42
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


def parse_q_grid(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        v = float(s)
        if v <= 0.0 or v >= 0.20:
            continue
        vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        vals = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]
    return vals


def parse_args():
    p = argparse.ArgumentParser(description="Train F1-push ranker v1")
    p.add_argument("--data_file", default="1_data/processed/train_hourly.csv")
    p.add_argument("--output_dir", default="4_models")
    p.add_argument(
        "--label_mode",
        choices=["official", "dynamic", "union"],
        default="union",
        help="official=official thresholds, dynamic=mean+1.5std, union=OR of both.",
    )
    p.add_argument(
        "--threshold_mat",
        default="1_data/raw/Seed_Coastal_Stations_Thresholds.mat",
        help="Path to Seed_Coastal_Stations_Thresholds.mat",
    )
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=200000,
        help="Cap per-fold train windows for speed; 0 disables cap.",
    )
    p.add_argument(
        "--q_grid",
        default="0.0005,0.001,0.002,0.003,0.005,0.0075,0.01",
        help="Comma-separated quantiles for flip policy search.",
    )
    p.add_argument(
        "--fn_penalty",
        type=float,
        default=1.2,
        help="Weighted-gain score: gain = TN - fn_penalty * FN",
    )
    p.add_argument(
        "--skip_save_model",
        action="store_true",
        help="Run CV policy search only; skip final fit.",
    )
    p.add_argument(
        "--weight_mode",
        choices=["spw", "balanced_prior", "station_balanced", "station_balanced_prior", "focal_cb"],
        default="spw",
        help=(
            "spw: classic scale_pos_weight. "
            "balanced_prior: class-prior-aware sample weights (A2). "
            "station_balanced: equalize station contribution. "
            "station_balanced_prior: combine both (B1). "
            "focal_cb: two-stage focal + class-balanced reweighting (A3)."
        ),
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal exponent gamma for focal_cb mode.",
    )
    p.add_argument(
        "--cb_beta",
        type=float,
        default=0.9999,
        help="Class-balanced beta (effective-number weighting) for focal_cb mode.",
    )
    p.add_argument(
        "--focal_w_clip_min",
        type=float,
        default=0.05,
        help="Lower clip for focal_cb sample weights before normalization.",
    )
    p.add_argument(
        "--focal_w_clip_max",
        type=float,
        default=20.0,
        help="Upper clip for focal_cb sample weights before normalization.",
    )
    p.add_argument(
        "--target_pos_rate",
        type=float,
        default=0.886724,
        help="Target positive prevalence for balanced_prior mode.",
    )
    p.add_argument(
        "--run_tag",
        default="f1push_ranker_v1",
        help="Run folder prefix under output_dir.",
    )
    p.add_argument(
        "--selection_mode",
        choices=["f1_first", "worst_station_mcc"],
        default="f1_first",
        help=(
            "f1_first: pooled gain/F1/MCC ranking (A1/A2). "
            "worst_station_mcc: prioritize worst-station robustness (B1)."
        ),
    )
    return p.parse_args()


def load_data(data_file: str) -> pd.DataFrame:
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


def create_daily_features(df: pd.DataFrame) -> pd.DataFrame:
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


def load_official_threshold_map(threshold_mat: str) -> dict[str, float]:
    mat_path = Path(threshold_mat)
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing threshold MAT: {mat_path}")
    mat = loadmat(mat_path)
    names = [x[0] for x in mat["sname"].squeeze()]
    vals = mat["thminor_stnd"].squeeze().astype(float)
    return {str(n): float(v) for n, v in zip(names, vals)}


def compute_dynamic_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    th = df.groupby("station_name")["sea_level"].agg(["mean", "std"]).reset_index()
    th["thr_dyn"] = th["mean"] + 1.5 * th["std"]
    return th[["station_name", "thr_dyn"]]


def attach_thresholds_and_relative_features(
    daily: pd.DataFrame,
    off_map: dict[str, float],
    dyn_df: pd.DataFrame,
) -> pd.DataFrame:
    x = daily.copy()
    x["thr_off"] = x["station_name"].map(off_map)
    x = x.merge(dyn_df, on="station_name", how="left")

    # Fallback for edge cases where official threshold is missing.
    x["thr_off"] = x["thr_off"].fillna(x["thr_dyn"])

    eps = 1e-6
    x["rel_mean_off"] = x["sea_level_mean"] - x["thr_off"]
    x["rel_max_off"] = x["sea_level_max"] - x["thr_off"]
    x["rel_3d_off"] = x["sea_level_3d_mean"] - x["thr_off"]
    x["rel_7d_off"] = x["sea_level_7d_mean"] - x["thr_off"]
    x["ratio_max_off"] = x["sea_level_max"] / (x["thr_off"].abs() + eps)

    st_stats = (
        x.groupby("station_name")["sea_level_mean"]
        .agg(st_mean="mean", st_std="std")
        .reset_index()
    )
    x = x.merge(st_stats, on="station_name", how="left")
    x["station_z_mean"] = (x["sea_level_mean"] - x["st_mean"]) / (x["st_std"].abs() + eps)
    x["station_rel_mean"] = x["sea_level_mean"] - x["st_mean"]

    return x.fillna(0.0)


def add_label_columns(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x["flood_day_off"] = (x["sea_level_max"] > x["thr_off"]).astype(np.int8)
    x["flood_day_dyn"] = (x["sea_level_max"] > x["thr_dyn"]).astype(np.int8)
    x["flood_day_union"] = np.maximum(x["flood_day_off"], x["flood_day_dyn"]).astype(np.int8)
    return x


def build_windows(
    daily: pd.DataFrame,
    stations: list[str],
    n_days: int,
    label_mode: str,
):
    if label_mode == "official":
        label_col = "flood_day_off"
    elif label_mode == "dynamic":
        label_col = "flood_day_dyn"
    else:
        label_col = "flood_day_union"

    rows_x, rows_y, rows_meta = [], [], []
    n_days = int(n_days)

    for stn in stations:
        g = daily[daily["station_name"] == stn].sort_values("date").reset_index(drop=True)
        if g.empty:
            continue

        feat = g[BASE_FEATURES].to_numpy(dtype=np.float32)
        flood = g[label_col].to_numpy(dtype=np.int8)
        dates = g["date"].to_numpy()

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

            y = int(flood[i + HIST_DAYS : i + HIST_DAYS + FUTURE_DAYS].max() > 0)
            rows_x.append(block.reshape(-1))
            rows_y.append(y)
            rows_meta.append(
                {
                    "station": stn,
                    "hist_end": pd.Timestamp(dates[i + HIST_DAYS - 1]).strftime("%Y-%m-%d"),
                }
            )

    if not rows_x:
        return (
            np.empty((0, len(BASE_FEATURES) * n_days), dtype=np.float32),
            np.empty((0,), dtype=np.int8),
            pd.DataFrame(columns=["station", "hist_end"]),
        )

    return np.asarray(rows_x, dtype=np.float32), np.asarray(rows_y, dtype=np.int8), pd.DataFrame(rows_meta)


def stratified_cap_indices(y: np.ndarray, cap: int, seed: int):
    n = int(len(y))
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return rng.choice(n, size=cap, replace=False)

    pos_rate = len(pos_idx) / len(y)
    n_pos = max(1, int(round(cap * pos_rate)))
    n_neg = max(1, cap - n_pos)
    s_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    s_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    sel = np.concatenate([s_pos, s_neg])
    rng.shuffle(sel)
    return sel.astype(np.int64)


def build_training_weights(
    y_train: np.ndarray,
    weight_mode: str,
    target_pos_rate: float,
    station_labels: Optional[np.ndarray] = None,
):
    pos_rate = float(y_train.mean()) if len(y_train) > 0 else 0.0
    info = {
        "weight_mode": str(weight_mode),
        "train_pos_rate": float(pos_rate),
        "target_pos_rate": float(target_pos_rate),
        "w_pos": 1.0,
        "w_neg": 1.0,
        "station_weight_min": 1.0,
        "station_weight_max": 1.0,
        "n_station_groups": 0,
    }

    if weight_mode == "spw" or len(y_train) == 0:
        return None, info

    sw = np.ones(len(y_train), dtype=np.float32)

    # A2: class-prior-aware weighting.
    if weight_mode in {"balanced_prior", "station_balanced_prior"}:
        eps = 1e-6
        p_src = float(np.clip(pos_rate, eps, 1.0 - eps))
        p_tgt = float(np.clip(target_pos_rate, eps, 1.0 - eps))
        w_pos = float(p_tgt / p_src)
        w_neg = float((1.0 - p_tgt) / (1.0 - p_src))
        sw *= np.where(y_train == 1, w_pos, w_neg).astype(np.float32)
        info["w_pos"] = float(w_pos)
        info["w_neg"] = float(w_neg)

    # B1: equalize per-station contribution.
    if weight_mode in {"station_balanced", "station_balanced_prior"}:
        if station_labels is None or len(station_labels) != len(y_train):
            raise ValueError("station_labels must align with y_train for station-balanced weighting.")
        vals, cnts = np.unique(station_labels.astype(str), return_counts=True)
        inv = {v: 1.0 / max(float(c), 1.0) for v, c in zip(vals, cnts)}
        st_w = np.asarray([inv[str(s)] for s in station_labels], dtype=np.float32)
        st_w /= float(np.mean(st_w))
        sw *= st_w
        info["station_weight_min"] = float(np.min(st_w))
        info["station_weight_max"] = float(np.max(st_w))
        info["n_station_groups"] = int(len(vals))

    sw /= float(np.mean(sw))
    return sw, info


def build_focal_cb_weights(
    y_train: np.ndarray,
    base_prob: np.ndarray,
    gamma: float,
    cb_beta: float,
    clip_min: float,
    clip_max: float,
):
    if len(y_train) == 0:
        info = {
            "weight_mode": "focal_cb",
            "train_pos_rate": 0.0,
            "target_pos_rate": 0.0,
            "w_pos": 1.0,
            "w_neg": 1.0,
            "station_weight_min": 1.0,
            "station_weight_max": 1.0,
            "n_station_groups": 0,
            "focal_gamma": float(gamma),
            "cb_beta": float(cb_beta),
            "focal_raw_min": 1.0,
            "focal_raw_max": 1.0,
        }
        return np.empty((0,), dtype=np.float32), info

    y = y_train.astype(np.int8)
    p = np.asarray(base_prob, dtype=np.float64).reshape(-1)
    if len(p) != len(y):
        raise ValueError("base_prob length must match y_train length for focal_cb mode.")
    p = np.clip(p, 1e-6, 1.0 - 1e-6)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_rate = float(y.mean())

    beta = float(np.clip(cb_beta, 0.0, 0.9999999))
    if beta <= 0.0:
        alpha_pos = 1.0
        alpha_neg = 1.0
    else:
        alpha_pos = float((1.0 - beta) / (1.0 - beta ** max(n_pos, 1)))
        alpha_neg = float((1.0 - beta) / (1.0 - beta ** max(n_neg, 1)))
        alpha_mean = (alpha_pos + alpha_neg) / 2.0
        alpha_pos /= max(alpha_mean, 1e-8)
        alpha_neg /= max(alpha_mean, 1e-8)

    pt = np.where(y == 1, p, 1.0 - p)
    focal = np.power(1.0 - pt, float(gamma))
    alpha = np.where(y == 1, alpha_pos, alpha_neg)
    w = alpha * focal

    lo = float(min(clip_min, clip_max))
    hi = float(max(clip_min, clip_max))
    w = np.clip(w, lo, hi)
    w /= float(np.mean(w))

    info = {
        "weight_mode": "focal_cb",
        "train_pos_rate": float(pos_rate),
        "target_pos_rate": float(pos_rate),
        "w_pos": float(alpha_pos),
        "w_neg": float(alpha_neg),
        "station_weight_min": 1.0,
        "station_weight_max": 1.0,
        "n_station_groups": 0,
        "focal_gamma": float(gamma),
        "cb_beta": float(beta),
        "focal_raw_min": float(np.min(w)),
        "focal_raw_max": float(np.max(w)),
    }
    return w.astype(np.float32), info


def make_clf(cfg: dict, scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        min_child_weight=float(cfg["min_child_weight"]),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=float(scale_pos_weight),
    )


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        return 0.5


def evaluate_quantile_policy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    q: float,
    fn_penalty: float,
) -> dict:
    n = int(len(y_true))
    if n == 0:
        return {
            "q": float(q),
            "k_flip": 0,
            "tn": 0,
            "fn": 0,
            "gain": 0.0,
            "neg_precision": 0.0,
            "auc": 0.5,
            "acc": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
        }

    k = max(1, int(n * q))
    low_idx = np.argpartition(y_prob, k - 1)[:k]

    y_pred = np.ones(n, dtype=np.int8)
    y_pred[low_idx] = 0

    tn = int((y_true[low_idx] == 0).sum())
    fn = int((y_true[low_idx] == 1).sum())
    neg_precision = float(tn / max(k, 1))
    gain = float(tn - fn_penalty * fn)

    return {
        "q": float(q),
        "k_flip": int(k),
        "tn": tn,
        "fn": fn,
        "gain": gain,
        "neg_precision": neg_precision,
        "auc": compute_auc(y_true, y_prob),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def get_configs(mode: str) -> list[dict]:
    quick = [
        {
            "name": "fr_q1",
            "n_days": 3,
            "n_estimators": 260,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
        },
        {
            "name": "fr_q2",
            "n_days": 2,
            "n_estimators": 260,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
        },
        {
            "name": "fr_q3",
            "n_days": 3,
            "n_estimators": 320,
            "max_depth": 5,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
        },
        {
            "name": "fr_q4",
            "n_days": 1,
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
        },
    ]

    full_extra = [
        {
            "name": "fr_f5",
            "n_days": 3,
            "n_estimators": 380,
            "max_depth": 5,
            "learning_rate": 0.035,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
        },
        {
            "name": "fr_f6",
            "n_days": 2,
            "n_estimators": 340,
            "max_depth": 5,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.5,
        },
        {
            "name": "fr_f7",
            "n_days": 1,
            "n_estimators": 320,
            "max_depth": 5,
            "learning_rate": 0.045,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
        },
    ]
    return quick if mode == "quick" else quick + full_extra


def run_cv_policy_search(
    daily: pd.DataFrame,
    stations: list[str],
    configs: list[dict],
    q_grid: list[float],
    label_mode: str,
    fn_penalty: float,
    max_train_samples: int,
    weight_mode: str,
    target_pos_rate: float,
    focal_gamma: float,
    cb_beta: float,
    focal_w_clip_min: float,
    focal_w_clip_max: float,
    selection_mode: str,
) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        print("\n" + "=" * 72)
        print(
            f"[cfg] {cfg['name']} | n_days={cfg['n_days']} "
            f"estimators={cfg['n_estimators']} depth={cfg['max_depth']} lr={cfg['learning_rate']}"
        )
        print("=" * 72)

        all_true = []
        all_prob = []
        fold_infos = []
        station_parts = []

        for i, test_station in enumerate(stations):
            train_stations = [s for s in stations if s != test_station]

            X_train, y_train, meta_train = build_windows(
                daily=daily,
                stations=train_stations,
                n_days=int(cfg["n_days"]),
                label_mode=label_mode,
            )
            X_test, y_test, _ = build_windows(
                daily=daily,
                stations=[test_station],
                n_days=int(cfg["n_days"]),
                label_mode=label_mode,
            )

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            cap_idx = stratified_cap_indices(y_train, cap=max_train_samples, seed=RANDOM_SEED + i)
            X_train = X_train[cap_idx]
            y_train = y_train[cap_idx]
            meta_train = meta_train.iloc[cap_idx].reset_index(drop=True)
            station_train = meta_train["station"].astype(str).to_numpy()

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            if weight_mode == "focal_cb":
                # Stage-1: baseline fit for hardness estimate.
                spw_stage1 = float(neg / max(pos, 1))
                clf_stage1 = make_clf(cfg, scale_pos_weight=spw_stage1)
                clf_stage1.fit(X_train_s, y_train, verbose=False)
                base_train_prob = clf_stage1.predict_proba(X_train_s)[:, 1]

                # Stage-2: focal + class-balanced reweighting.
                sample_w, w_info = build_focal_cb_weights(
                    y_train=y_train,
                    base_prob=base_train_prob,
                    gamma=float(focal_gamma),
                    cb_beta=float(cb_beta),
                    clip_min=float(focal_w_clip_min),
                    clip_max=float(focal_w_clip_max),
                )
                clf = make_clf(cfg, scale_pos_weight=1.0)
                clf.fit(X_train_s, y_train, sample_weight=sample_w, verbose=False)
            else:
                if weight_mode == "spw":
                    spw = float(neg / max(pos, 1))
                else:
                    spw = 1.0
                sample_w, w_info = build_training_weights(
                    y_train=y_train,
                    weight_mode=weight_mode,
                    target_pos_rate=float(target_pos_rate),
                    station_labels=station_train,
                )

                clf = make_clf(cfg, scale_pos_weight=spw)
                if sample_w is None:
                    clf.fit(X_train_s, y_train, verbose=False)
                else:
                    clf.fit(X_train_s, y_train, sample_weight=sample_w, verbose=False)
            y_prob = clf.predict_proba(X_test_s)[:, 1]

            auc = compute_auc(y_test, y_prob)
            if weight_mode == "focal_cb":
                w_msg = (
                    f"alpha_pos={w_info['w_pos']:.3f} alpha_neg={w_info['w_neg']:.3f} "
                    f"gamma={w_info['focal_gamma']:.2f}"
                )
            else:
                w_msg = f"w_pos={w_info['w_pos']:.3f} w_neg={w_info['w_neg']:.3f}"
            print(
                f"  {test_station:20s}: n={len(y_test):6d} pos={y_test.mean():.4f} auc={auc:.4f} "
                f"{w_msg}"
            )

            all_true.append(y_test)
            all_prob.append(y_prob)
            fold_infos.append(w_info)
            station_parts.append(
                {
                    "station": str(test_station),
                    "y_true": y_test,
                    "y_prob": y_prob,
                }
            )

        if not all_true:
            continue

        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_prob)
        mean_train_pos = float(np.mean([x["train_pos_rate"] for x in fold_infos])) if fold_infos else 0.0
        mean_w_pos = float(np.mean([x["w_pos"] for x in fold_infos])) if fold_infos else 1.0
        mean_w_neg = float(np.mean([x["w_neg"] for x in fold_infos])) if fold_infos else 1.0

        base_auc = compute_auc(y_true, y_prob)
        print(f"[summary] pooled_n={len(y_true):,} pooled_pos={y_true.mean():.4f} auc={base_auc:.4f}")

        for q in q_grid:
            m = evaluate_quantile_policy(y_true=y_true, y_prob=y_prob, q=q, fn_penalty=fn_penalty)
            st_ms = [
                (
                    part["station"],
                    evaluate_quantile_policy(
                        y_true=part["y_true"],
                        y_prob=part["y_prob"],
                        q=q,
                        fn_penalty=fn_penalty,
                    ),
                )
                for part in station_parts
            ]
            worst_station_name, worst_station_metrics = min(st_ms, key=lambda t: t[1]["mcc"])
            worst_station_mcc = float(worst_station_metrics["mcc"])
            worst_station_f1 = float(worst_station_metrics["f1"])
            mean_station_mcc = float(np.mean([x[1]["mcc"] for x in st_ms])) if st_ms else 0.0
            print(
                f"  q={q:.4f} k={m['k_flip']:4d} tn={m['tn']:4d} fn={m['fn']:4d} "
                f"gain={m['gain']:.1f} f1={m['f1']:.4f} mcc={m['mcc']:.4f} "
                f"worst_mcc={worst_station_mcc:.4f} ({worst_station_name})"
            )
            rows.append(
                {
                    "config_name": cfg["name"],
                    "n_days": int(cfg["n_days"]),
                    "n_estimators": int(cfg["n_estimators"]),
                    "max_depth": int(cfg["max_depth"]),
                    "learning_rate": float(cfg["learning_rate"]),
                    "subsample": float(cfg["subsample"]),
                    "colsample_bytree": float(cfg["colsample_bytree"]),
                    "min_child_weight": float(cfg["min_child_weight"]),
                    "q": float(q),
                    "fn_penalty": float(fn_penalty),
                    "pooled_n": int(len(y_true)),
                    "pooled_pos_rate": float(y_true.mean()),
                    "pooled_auc": float(base_auc),
                    "weight_mode": str(weight_mode),
                    "target_pos_rate": float(target_pos_rate),
                    "focal_gamma": float(focal_gamma),
                    "cb_beta": float(cb_beta),
                    "mean_train_pos_rate": float(mean_train_pos),
                    "mean_w_pos": float(mean_w_pos),
                    "mean_w_neg": float(mean_w_neg),
                    "worst_station_name": str(worst_station_name),
                    "worst_station_mcc": float(worst_station_mcc),
                    "worst_station_f1": float(worst_station_f1),
                    "mean_station_mcc": float(mean_station_mcc),
                    "selection_mode": str(selection_mode),
                    **m,
                }
            )

    if not rows:
        raise RuntimeError("No valid CV rows produced.")

    out = pd.DataFrame(rows)
    if selection_mode == "worst_station_mcc":
        out = out.sort_values(
            ["worst_station_mcc", "mcc", "f1", "gain", "pooled_auc"],
            ascending=False,
        )
    else:
        out = out.sort_values(
            ["gain", "f1", "mcc", "neg_precision", "pooled_auc"],
            ascending=False,
        )
    return out.reset_index(drop=True)


def train_final_model(
    daily: pd.DataFrame,
    stations: list[str],
    cfg: dict,
    label_mode: str,
    weight_mode: str,
    target_pos_rate: float,
    focal_gamma: float,
    cb_beta: float,
    focal_w_clip_min: float,
    focal_w_clip_max: float,
):
    X_all, y_all, meta_all = build_windows(
        daily=daily,
        stations=stations,
        n_days=int(cfg["n_days"]),
        label_mode=label_mode,
    )
    if len(X_all) == 0:
        raise RuntimeError("No windows for final fit.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)
    pos = int((y_all == 1).sum())
    neg = int((y_all == 0).sum())
    if weight_mode == "focal_cb":
        spw_stage1 = float(neg / max(pos, 1))
        clf_stage1 = make_clf(cfg, scale_pos_weight=spw_stage1)
        clf_stage1.fit(Xs, y_all, verbose=False)
        base_prob = clf_stage1.predict_proba(Xs)[:, 1]
        sample_w, w_info = build_focal_cb_weights(
            y_train=y_all,
            base_prob=base_prob,
            gamma=float(focal_gamma),
            cb_beta=float(cb_beta),
            clip_min=float(focal_w_clip_min),
            clip_max=float(focal_w_clip_max),
        )
        clf = make_clf(cfg, scale_pos_weight=1.0)
        clf.fit(Xs, y_all, sample_weight=sample_w, verbose=False)
        spw = 1.0
    else:
        if weight_mode == "spw":
            spw = float(neg / max(pos, 1))
        else:
            spw = 1.0
        sample_w, w_info = build_training_weights(
            y_train=y_all,
            weight_mode=weight_mode,
            target_pos_rate=float(target_pos_rate),
            station_labels=meta_all["station"].astype(str).to_numpy(),
        )
        clf = make_clf(cfg, scale_pos_weight=spw)
        if sample_w is None:
            clf.fit(Xs, y_all, verbose=False)
        else:
            clf.fit(Xs, y_all, sample_weight=sample_w, verbose=False)

    return {
        "model": clf,
        "scaler": scaler,
        "n_train": int(len(X_all)),
        "flood_rate": float(y_all.mean()),
        "scale_pos_weight": float(spw),
        "weight_mode": str(weight_mode),
        "target_pos_rate": float(target_pos_rate),
        "w_pos": float(w_info["w_pos"]),
        "w_neg": float(w_info["w_neg"]),
        "focal_gamma": float(w_info.get("focal_gamma", focal_gamma)),
        "cb_beta": float(w_info.get("cb_beta", cb_beta)),
    }


def write_run_note(out_dir: Path, selected: dict):
    lines = [
        "# F1-Push Ranker Run Note",
        "",
        f"- run_dir: `{out_dir}`",
        f"- selected_config: `{selected['config_name']}`",
        f"- selected_q: `{selected['q']}`",
        f"- selected_gain: `{selected['gain']}`",
        f"- selected_f1: `{selected['f1']}`",
        f"- selected_mcc: `{selected['mcc']}`",
        "",
    ]
    (out_dir / "run_note.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    q_grid = parse_q_grid(args.q_grid)
    start = datetime.now()

    print("=" * 72)
    print("F1-Push Ranker Training Suite v1")
    print(f"label_mode={args.label_mode}")
    print(f"threshold_mat={args.threshold_mat}")
    print(f"mode={args.mode}")
    print(f"data_file={args.data_file}")
    print(f"output_root={args.output_dir}")
    print(f"max_train_samples={args.max_train_samples}")
    print(f"fn_penalty={args.fn_penalty}")
    print(f"weight_mode={args.weight_mode}")
    print(f"target_pos_rate={args.target_pos_rate}")
    print(f"focal_gamma={args.focal_gamma}")
    print(f"cb_beta={args.cb_beta}")
    print(f"focal_w_clip=({args.focal_w_clip_min},{args.focal_w_clip_max})")
    print(f"run_tag={args.run_tag}")
    print(f"selection_mode={args.selection_mode}")
    print(f"q_grid={q_grid}")
    print("=" * 72)

    train_df = load_data(args.data_file)
    daily = create_daily_features(train_df)
    stations = sorted(daily["station_name"].unique().tolist())

    off_map = load_official_threshold_map(args.threshold_mat)
    dyn_df = compute_dynamic_thresholds(train_df)
    daily = attach_thresholds_and_relative_features(daily, off_map=off_map, dyn_df=dyn_df)
    daily = add_label_columns(daily)

    print(f"rows={len(train_df):,} stations={stations}")
    print(
        "day_pos_rate:",
        {
            "official": float(daily["flood_day_off"].mean()),
            "dynamic": float(daily["flood_day_dyn"].mean()),
            "union": float(daily["flood_day_union"].mean()),
        },
    )

    run_id = datetime.now().strftime(f"{args.run_tag}_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = get_configs(args.mode)
    cv_df = run_cv_policy_search(
        daily=daily,
        stations=stations,
        configs=configs,
        q_grid=q_grid,
        label_mode=args.label_mode,
        fn_penalty=float(args.fn_penalty),
        max_train_samples=int(args.max_train_samples),
        weight_mode=args.weight_mode,
        target_pos_rate=float(args.target_pos_rate),
        focal_gamma=float(args.focal_gamma),
        cb_beta=float(args.cb_beta),
        focal_w_clip_min=float(args.focal_w_clip_min),
        focal_w_clip_max=float(args.focal_w_clip_max),
        selection_mode=args.selection_mode,
    )
    cv_df.to_csv(out_dir / "cv_policy_results.csv", index=False)

    best = cv_df.iloc[0].to_dict()
    selected_cfg = {
        "name": str(best["config_name"]),
        "n_days": int(best["n_days"]),
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "min_child_weight": float(best["min_child_weight"]),
    }

    print("\n" + "=" * 72)
    print(f"[selected] cfg={selected_cfg} q={best['q']} gain={best['gain']:.4f}")
    print("=" * 72)

    model_meta = None
    if args.skip_save_model:
        print("skip_save_model=True -> skip final fit")
    else:
        fit = train_final_model(
            daily=daily,
            stations=stations,
            cfg=selected_cfg,
            label_mode=args.label_mode,
            weight_mode=args.weight_mode,
            target_pos_rate=float(args.target_pos_rate),
            focal_gamma=float(args.focal_gamma),
            cb_beta=float(args.cb_beta),
            focal_w_clip_min=float(args.focal_w_clip_min),
            focal_w_clip_max=float(args.focal_w_clip_max),
        )
        bundle = {
            "model_kind": "f1push_ranker_v1",
            "model": fit["model"],
            "scaler": fit["scaler"],
            "feature_cols": BASE_FEATURES,
            "use_last_n_days": int(selected_cfg["n_days"]),
            "selected_q": float(best["q"]),
            "high_prob_value": 0.99,
            "low_prob_value": 0.49,
            "label_mode": args.label_mode,
            "official_threshold_map": {str(k): float(v) for k, v in off_map.items()},
            "generated_at": datetime.now().isoformat(),
            "weight_mode": args.weight_mode,
            "target_pos_rate": float(args.target_pos_rate),
            "focal_gamma": float(args.focal_gamma),
            "cb_beta": float(args.cb_beta),
        }
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(bundle, f)

        model_meta = {
            "selected_config": selected_cfg,
            "selected_q": float(best["q"]),
            "train_samples": int(fit["n_train"]),
            "flood_rate": float(fit["flood_rate"]),
            "scale_pos_weight": float(fit["scale_pos_weight"]),
            "weight_mode": str(fit["weight_mode"]),
            "target_pos_rate": float(fit["target_pos_rate"]),
            "w_pos": float(fit["w_pos"]),
            "w_neg": float(fit["w_neg"]),
            "focal_gamma": float(fit["focal_gamma"]),
            "cb_beta": float(fit["cb_beta"]),
            "label_mode": args.label_mode,
            "threshold_source": "official_mat+dynamic",
            "generated_at": datetime.now().isoformat(),
            "model_kind": "f1push_ranker_v1",
        }
        with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2)

    results = {
        "run_dir": str(out_dir),
        "label_mode": args.label_mode,
        "threshold_source": "official_mat+dynamic",
        "fn_penalty": float(args.fn_penalty),
        "weight_mode": args.weight_mode,
        "target_pos_rate": float(args.target_pos_rate),
        "focal_gamma": float(args.focal_gamma),
        "cb_beta": float(args.cb_beta),
        "run_tag": args.run_tag,
        "selection_mode": args.selection_mode,
        "q_grid": q_grid,
        "selected_row": best,
        "selected_config": selected_cfg,
        "skip_save_model": bool(args.skip_save_model),
    }
    if model_meta is not None:
        results["model_meta"] = model_meta

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_run_note(out_dir=out_dir, selected=best)

    elapsed = datetime.now() - start
    print("\n" + "=" * 72)
    print(f"Done. run_dir={out_dir}")
    print(f"Elapsed={elapsed}")
    print("=" * 72)


if __name__ == "__main__":
    main()
