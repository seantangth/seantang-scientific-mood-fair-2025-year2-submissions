#!/usr/bin/env python3
"""
Create D2 station-quota consensus submission ZIPs.

Why D2:
- D1 still used global tiny-q, which can keep selecting nearly identical rows.
- D2 allocates flips by station with learned station negative priors, then selects
  lowest-risk rows inside each station.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

HIST_DAYS = 7
FUTURE_DAYS = 14


MODEL_TEMPLATE = r"""#!/usr/bin/env python3
import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HIST_DAYS = 7
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

FORCE_LOWEST_Q_TO_NEG = __FORCE_Q__
HIGH_PROB_VALUE = __HIGH_PROB__
LOW_PROB_VALUE = __LOW_PROB__
STATION_NEG_PRIOR = __STATION_NEG_PRIOR__
NEG_PRIOR_ALPHA = __NEG_PRIOR_ALPHA__


def daily_aggregate(df):
    x = df.copy()
    x["time"] = pd.to_datetime(x["time"])
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


def attach_relative_features(daily, threshold_map):
    x = daily.copy()
    x["thr_off"] = x["station_name"].map(threshold_map)

    dyn = x.groupby("station_name")["sea_level_mean"].agg(["mean", "std"]).reset_index()
    dyn["thr_dyn"] = dyn["mean"] + 1.5 * dyn["std"]
    x = x.merge(dyn[["station_name", "thr_dyn"]], on="station_name", how="left")
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


def build_feature_table(daily, use_last_n_days):
    rows_x, keys = [], []
    use_last_n_days = int(use_last_n_days)

    for station, grp in daily.groupby("station_name"):
        g = grp.sort_values("date").reset_index(drop=True)
        feat = g[BASE_FEATURES].to_numpy(dtype=np.float32)
        dates = g["date"].dt.strftime("%Y-%m-%d").to_numpy()

        for i in range(HIST_DAYS - 1, len(g)):
            start_idx = i - use_last_n_days + 1
            if start_idx < 0:
                continue
            block = feat[start_idx : i + 1]
            if block.shape[0] != use_last_n_days:
                continue
            rows_x.append(block.reshape(-1))
            keys.append(f"{station}|{dates[i]}")

    if not rows_x:
        return np.empty((0, len(BASE_FEATURES) * use_last_n_days), dtype=np.float32), pd.DataFrame(columns=["key"])
    return np.asarray(rows_x, dtype=np.float32), pd.DataFrame({"key": keys})


def score_bundle(bundle, daily, index_df):
    use_last_n_days = int(bundle.get("use_last_n_days", 3))
    X_test, meta = build_feature_table(daily, use_last_n_days=use_last_n_days)
    if len(X_test) == 0:
        return np.full(len(index_df), 0.5, dtype=np.float32)

    model = bundle["model"]
    scaler = bundle["scaler"]
    Xs = scaler.transform(X_test)
    if hasattr(model, "predict_proba"):
        flood_prob = model.predict_proba(Xs)[:, 1].astype(np.float32)
    else:
        flood_prob = np.asarray(model.predict(Xs), dtype=np.float32).reshape(-1)
        flood_prob = np.clip(flood_prob, 0.0, 1.0)

    pred = meta.copy()
    pred["flood_prob"] = flood_prob
    out = index_df.merge(pred, on="key", how="left")[["id", "flood_prob"]]
    out["flood_prob"] = out["flood_prob"].fillna(0.5).clip(0.0, 1.0)
    return out["flood_prob"].to_numpy(dtype=np.float32)


def consensus_risk(prob_matrix):
    # Lower score means safer / more likely non-flood.
    n_models, n = prob_matrix.shape
    rank_mat = np.zeros((n_models, n), dtype=np.float32)
    for i in range(n_models):
        order = np.argsort(prob_matrix[i])
        rk = np.empty(n, dtype=np.int64)
        rk[order] = np.arange(n)
        rank_mat[i] = rk / max(n - 1, 1)
    return rank_mat.mean(axis=0)


def allocate_station_ks(stations, k_total, station_neg_prior, alpha):
    if k_total <= 0:
        return {}
    st = stations.astype(str)
    uniq, cnt = np.unique(st, return_counts=True)
    if len(uniq) == 0:
        return {}

    global_prior = float(np.mean(list(station_neg_prior.values()))) if station_neg_prior else 0.5
    weights = []
    for u, c in zip(uniq, cnt):
        p_neg = float(station_neg_prior.get(str(u), global_prior))
        p_neg = min(max(p_neg, 1e-6), 1.0 - 1e-6)
        w = (p_neg ** float(alpha)) * float(c)
        weights.append(w)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)

    target = weights * float(k_total)
    ks = np.floor(target).astype(int)
    ks = np.minimum(ks, cnt.astype(int))
    rem = int(k_total - ks.sum())

    frac = target - ks
    order = np.argsort(-frac)
    while rem > 0:
        moved = False
        for j in order:
            if rem <= 0:
                break
            if ks[j] < int(cnt[j]):
                ks[j] += 1
                rem -= 1
                moved = True
        if not moved:
            break
    return {str(u): int(k) for u, k in zip(uniq, ks)}


def station_quota_selection(index_df, risk_score, q, station_neg_prior, alpha):
    n = len(risk_score)
    if n == 0:
        return np.array([], dtype=np.int64)
    if q <= 0.0:
        return np.array([], dtype=np.int64)
    k_total = max(1, int(n * q))

    stations = index_df["station_name"].astype(str).to_numpy()
    k_map = allocate_station_ks(stations, k_total, station_neg_prior, alpha)
    chosen = []
    for stn, k in k_map.items():
        if k <= 0:
            continue
        idx = np.where(stations == stn)[0]
        if len(idx) == 0:
            continue
        order_local = idx[np.argsort(risk_score[idx])]
        chosen.append(order_local[: min(k, len(order_local))])
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.concatenate(chosen).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", required=True)
    ap.add_argument("--test_hourly", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--predictions_out", required=True)
    args = ap.parse_args()

    model_dir = Path(__file__).resolve().parent
    bundle_paths = sorted(glob.glob(str(model_dir / "bundle_*.pkl")))
    if not bundle_paths:
        raise RuntimeError("No bundle_*.pkl found in submission folder.")

    bundles = []
    threshold_map = {}
    for bp in bundle_paths:
        with open(bp, "rb") as f:
            b = pickle.load(f)
        if str(b.get("model_kind", "")) != "f1push_ranker_v1":
            continue
        bundles.append(b)
        if not threshold_map:
            threshold_map = dict(b.get("official_threshold_map", {}))
    if not bundles:
        raise RuntimeError("No valid f1push bundles loaded.")

    test_hourly = pd.read_csv(args.test_hourly)
    index = pd.read_csv(args.test_index)
    index["hist_end"] = pd.to_datetime(index["hist_end"]).dt.strftime("%Y-%m-%d")
    index["key"] = index["station_name"].astype(str) + "|" + index["hist_end"].astype(str)

    daily = daily_aggregate(test_hourly)
    daily = attach_relative_features(daily, threshold_map=threshold_map)

    prob_rows = []
    for b in bundles:
        prob_rows.append(score_bundle(b, daily=daily, index_df=index))
    prob_matrix = np.asarray(prob_rows, dtype=np.float32)
    risk = consensus_risk(prob_matrix=prob_matrix)

    n = len(index)
    order = np.argsort(risk)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    rank_norm = rank / max(n - 1, 1)
    y_prob = (HIGH_PROB_VALUE - 0.02 * (1.0 - rank_norm)).astype(np.float32)

    low_idx = station_quota_selection(
        index_df=index,
        risk_score=risk,
        q=float(FORCE_LOWEST_Q_TO_NEG),
        station_neg_prior=STATION_NEG_PRIOR,
        alpha=float(NEG_PRIOR_ALPHA),
    )
    if len(low_idx) > 0:
        y_prob[low_idx] = LOW_PROB_VALUE

    out = index[["id"]].copy()
    out["y_prob"] = np.clip(y_prob, 0.0, 1.0)
    out.to_csv(args.predictions_out, index=False)


if __name__ == "__main__":
    main()
"""


def parse_args():
    p = argparse.ArgumentParser(description="Create D2 station-quota consensus submission zips")
    p.add_argument(
        "--run_dirs",
        required=True,
        help="Comma-separated run dirs (f1push bundles) under 4_models or absolute paths.",
    )
    p.add_argument("--train_hourly", default="1_data/processed/train_hourly.csv")
    p.add_argument("--threshold_mat", default="1_data/raw/Seed_Coastal_Stations_Thresholds.mat")
    p.add_argument("--output_root", default="5_outputs/submissions")
    p.add_argument("--prefix", default="d2s2")
    p.add_argument("--q_list", default="0.0,0.00001,0.00002,0.00005,0.00010,0.00020,0.00050")
    p.add_argument("--alpha", type=float, default=2.0, help="Exponent over station negative prior.")
    p.add_argument("--high_prob", type=float, default=0.99)
    p.add_argument("--low_prob", type=float, default=0.49)
    return p.parse_args()


def parse_q_list(text: str) -> list[float]:
    vals = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        v = float(s)
        if v < 0.0 or v >= 0.20:
            continue
        vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        vals = [0.0, 0.00001, 0.00002, 0.00005]
    return vals


def q_slug(q: float) -> str:
    if q <= 0:
        return "base"
    return f"q{int(round(q * 100000)):05d}"


def load_official_threshold_map(threshold_mat: str) -> dict[str, float]:
    mat = loadmat(threshold_mat)
    names = [x[0] for x in mat["sname"].squeeze()]
    vals = mat["thminor_stnd"].squeeze().astype(float)
    return {str(n): float(v) for n, v in zip(names, vals)}


def estimate_station_neg_prior(train_hourly: str, threshold_mat: str) -> dict[str, float]:
    df = pd.read_csv(train_hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.floor("D")
    off_map = load_official_threshold_map(threshold_mat)

    daily = (
        df.groupby(["station_name", "date"])
        .agg(sea_level_max=("sea_level", "max"))
        .reset_index()
        .sort_values(["station_name", "date"])
        .reset_index(drop=True)
    )
    daily["thr_off"] = daily["station_name"].map(off_map)
    if daily["thr_off"].isna().any():
        dyn = df.groupby("station_name")["sea_level"].agg(["mean", "std"]).reset_index()
        dyn["thr_dyn"] = dyn["mean"] + 1.5 * dyn["std"]
        daily = daily.merge(dyn[["station_name", "thr_dyn"]], on="station_name", how="left")
        daily["thr_off"] = daily["thr_off"].fillna(daily["thr_dyn"])
    daily["flood_day_off"] = (daily["sea_level_max"] > daily["thr_off"]).astype(np.int8)

    neg_prior = {}
    for stn, g in daily.groupby("station_name"):
        g = g.sort_values("date").reset_index(drop=True)
        flood = g["flood_day_off"].to_numpy(dtype=np.int8)
        ys = []
        max_i = len(g) - HIST_DAYS - FUTURE_DAYS + 1
        for i in range(max_i):
            y = int(flood[i + HIST_DAYS : i + HIST_DAYS + FUTURE_DAYS].max() > 0)
            ys.append(y)
        if ys:
            pos_rate = float(np.mean(ys))
            neg_prior[str(stn)] = float(1.0 - pos_rate)
        else:
            neg_prior[str(stn)] = 0.5
    return neg_prior


def main():
    args = parse_args()
    run_dirs = [x.strip() for x in args.run_dirs.split(",") if x.strip()]
    if not run_dirs:
        raise ValueError("Empty --run_dirs")

    source_model_paths = []
    source_names = []
    for rd in run_dirs:
        p = Path(rd)
        if not p.exists():
            p = Path("4_models") / rd
        mp = p / "model.pkl"
        if not mp.exists():
            raise FileNotFoundError(f"Missing model.pkl in run dir: {p}")
        with open(mp, "rb") as f:
            b = pickle.load(f)
        if str(b.get("model_kind", "")) != "f1push_ranker_v1":
            raise RuntimeError(f"Unsupported model_kind in {p}: {b.get('model_kind')}")
        source_model_paths.append(mp)
        source_names.append(p.name)

    station_neg_prior = estimate_station_neg_prior(
        train_hourly=args.train_hourly,
        threshold_mat=args.threshold_mat,
    )
    print("[info] station_neg_prior:", station_neg_prior)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    q_vals = parse_q_list(args.q_list)
    created = []
    for q in q_vals:
        slug = q_slug(q)
        folder_name = f"{args.prefix}_{slug}_v1"
        dst = out_root / folder_name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        for i, mp in enumerate(source_model_paths):
            shutil.copy2(mp, dst / f"bundle_{i:02d}.pkl")

        tpl = MODEL_TEMPLATE
        tpl = tpl.replace("__FORCE_Q__", repr(float(q)))
        tpl = tpl.replace("__HIGH_PROB__", repr(float(args.high_prob)))
        tpl = tpl.replace("__LOW_PROB__", repr(float(args.low_prob)))
        tpl = tpl.replace("__NEG_PRIOR_ALPHA__", repr(float(args.alpha)))
        tpl = tpl.replace("__STATION_NEG_PRIOR__", repr(station_neg_prior))
        (dst / "model.py").write_text(tpl, encoding="utf-8")

        meta = {
            "model_kind": "station_quota_consensus_f1push_v1",
            "sources": source_names,
            "submission_q": float(q),
            "station_neg_prior": station_neg_prior,
            "neg_prior_alpha": float(args.alpha),
            "high_prob_value": float(args.high_prob),
            "low_prob_value": float(args.low_prob),
        }
        (dst / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (dst / "requirements.txt").write_text(
            "\n".join(["numpy", "pandas", "scikit-learn", "xgboost"]) + "\n",
            encoding="utf-8",
        )
        (dst / "README.md").write_text(
            "\n".join(
                [
                    "# D2 station-quota consensus submission",
                    "",
                    f"- source_runs: `{', '.join(source_names)}`",
                    f"- force_lowest_q_to_neg: `{q}`",
                    f"- neg_prior_alpha: `{args.alpha}`",
                    "- policy: station-quota consensus low-risk selection",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        zip_path = shutil.make_archive(str(dst), "zip", root_dir=dst)
        created.append(zip_path)
        print(f"[ok] {zip_path}")

    print("\nCreated submission zips:")
    for z in created:
        print(f"- {z}")


if __name__ == "__main__":
    main()

