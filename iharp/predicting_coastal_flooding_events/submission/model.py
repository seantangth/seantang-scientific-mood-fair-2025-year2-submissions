#!/usr/bin/env python3
"""
Hybrid-Final: d2s2 + G7v2 Ensemble for Per-Window, G7v2-only for Per-Day
=========================================================================
Per-window mode: rank-based ensemble (0.6 * d2s2_rank + 0.4 * g7v2_rank)
Per-day mode: G7v2 with corrected day_offset only (d2s2 not suited for per-day)
"""
import argparse
import glob as _glob
import json
import math
import pickle
import time as _time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception:
    xgb = None

# ======================================================================
# G7v2 constants
# ======================================================================
HIST_DAYS = 7
FUTURE_DAYS = 14
N_HARMONICS = 5
FEAT_DIM = 62

_SN_REF_SEC = 947189640.0
SPRING_NEAP_PERIOD = 29.53059 / 2.0
_2PI = 2.0 * math.pi

OFFICIAL_THRESHOLDS = {
    "Annapolis": 2.104, "Atlantic_City": 3.344, "Charleston": 2.98,
    "Eastport": 8.071, "Fernandina_Beach": 3.148, "Lewes": 2.675,
    "Portland": 6.267, "Sandy_Hook": 2.809, "Sewells_Point": 2.706,
    "The_Battery": 3.192, "Washington": 2.673, "Wilmington": 2.423,
}

# ======================================================================
# d2s2 constants
# ======================================================================
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

FORCE_LOWEST_Q_TO_NEG = 1e-5
HIGH_PROB_VALUE = 0.99
LOW_PROB_VALUE = 0.49
STATION_NEG_PRIOR = {
    'Annapolis': 0.6814340292517269,
    'Atlantic_City': 0.7946590514413615,
    'Charleston': 0.7927681086713233,
    'Eastport': 0.7030448037664493,
    'Portland': 0.9146374406668467,
    'Sandy_Hook': 0.7887160884498128,
    'Sewells_Point': 0.9082699803187589,
    'Washington': 0.7290549145216687,
    'Wilmington': 0.7600432215490295,
}
NEG_PRIOR_ALPHA = 2.0


# ======================================================================
# d2s2 functions
# ======================================================================
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
    daily["sea_level_diff_1d"] = daily.groupby("station_name")["sea_level_mean"].transform(
        lambda s: s.diff(1)
    )
    daily["sea_level_diff_3d"] = daily.groupby("station_name")["sea_level_mean"].transform(
        lambda s: s.diff(3)
    )
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


# ======================================================================
# G7v2 functions
# ======================================================================
def max_consecutive_above(values, threshold):
    n = len(values)
    if n == 0:
        return 0
    above = values > threshold
    if not above.any():
        return 0
    d = np.diff(np.concatenate(([False], above, [False])).astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return int((ends - starts).max()) if len(starts) > 0 else 0


def compute_tidal_info(sl, threshold):
    n = len(sl)
    harmonics = []
    tidal_mean = float(np.mean(sl)) if n > 0 else 0.0
    res_feats = np.zeros(4, dtype=np.float32)
    last_exc = -1

    if n > 0:
        above = sl > threshold
        if above.any():
            last_exc = int(np.where(above)[0][-1])

    if n < 48:
        return harmonics, tidal_mean, n, res_feats, last_exc

    mean_lev = float(np.mean(sl))
    fft_v = np.fft.rfft(sl - mean_lev)
    freqs = np.fft.rfftfreq(n, d=1.0)
    amps = np.abs(fft_v) * 2.0 / n
    phases = np.angle(fft_v)

    amp_s = amps.copy()
    amp_s[0] = 0.0
    amp_s[freqs < 1.0 / float(n)] = 0.0
    for idx in np.argsort(amp_s)[-N_HARMONICS:]:
        if amp_s[idx] > 1e-8:
            harmonics.append((float(freqs[idx]), float(amps[idx]), float(phases[idx])))

    t_h = np.arange(n, dtype=np.float64)
    fit = np.full(n, mean_lev, dtype=np.float64)
    for f, a, p in harmonics:
        fit += a * np.cos(_2PI * f * t_h + p)
    res = sl - fit

    r_mean = float(np.mean(res))
    r_std = float(np.std(res))
    nd = n // 24
    if nd >= 2:
        dr = np.array([float(np.mean(res[i*24:(i+1)*24])) for i in range(nd)])
        r_trend = float(np.polyfit(np.arange(nd, dtype=np.float64), dr, 1)[0])
    else:
        r_trend = 0.0
    r_last = float(np.mean(res[-min(24, n):]))
    res_feats = np.array([r_mean, r_std, r_trend, r_last], dtype=np.float32)
    return harmonics, mean_lev, n, res_feats, last_exc


def build_window_features(
    h_mean, h_max, h_min, h_std,
    r3m, r7m, r3s, r7s,
    hourly_sl, thr, stn_mean, stn_std,
    n_days, future_doy, future_epoch_sec,
    harmonics, tidal_mean, n_hist_hours,
    res_feats, last_exc,
    base_day_offset=1,
):
    feats = np.zeros((n_days, FEAT_DIM), dtype=np.float32)
    eps = 1e-8

    mean_rel = h_mean - thr
    max_rel = h_max - thr
    min_rel = h_min - thr

    feats[:, 0:7] = mean_rel
    feats[:, 7:14] = max_rel
    feats[:, 14:21] = min_rel
    feats[:, 21:28] = h_std
    feats[:, 28] = r3m - thr
    feats[:, 29] = r7m - thr
    feats[:, 30] = r3s
    feats[:, 31] = r7s

    n_h = len(hourly_sl)
    feats[:, 32] = float(np.mean(hourly_sl > thr)) if n_h > 0 else 0.0
    feats[:, 33] = float(max_consecutive_above(hourly_sl, thr))

    feats[:, 34] = float(mean_rel[-1] - mean_rel[-3]) if len(mean_rel) >= 3 else 0.0
    if len(mean_rel) >= 2:
        feats[:, 35] = float(np.polyfit(np.arange(len(mean_rel), dtype=np.float32), mean_rel, 1)[0])

    feats[:, 36] = float(np.mean(max_rel))
    feats[:, 37] = float(np.max(max_rel))
    feats[:, 38] = float(np.std(max_rel))

    feats[:, 39:46] = (h_mean - stn_mean) / (stn_std + eps)

    # Temporal (per-day) -- CORRECTED: use base_day_offset
    day_offsets = np.arange(base_day_offset, base_day_offset + n_days, dtype=np.float32)
    feats[:, 46] = day_offsets / float(FUTURE_DAYS)
    ang = _2PI * future_doy / 366.0
    feats[:, 47] = np.sin(ang)
    feats[:, 48] = np.cos(ang)

    # Tidal projection (vectorized) -- CORRECTED: start from correct hour offset
    if harmonics and n_hist_hours > 0:
        start_hour = n_hist_hours + (base_day_offset - 1) * 24
        all_h = np.arange(start_hour, start_hour + n_days * 24, dtype=np.float64)
        sig = np.full(n_days * 24, tidal_mean, dtype=np.float64)
        for f, a, p in harmonics:
            sig += a * np.cos(_2PI * f * all_h + p)
        s2d = sig.reshape(n_days, 24)
        feats[:, 49] = s2d.max(axis=1) - thr
        feats[:, 50] = s2d.min(axis=1) - thr
        feats[:, 51] = s2d.max(axis=1) - s2d.min(axis=1)
        feats[:, 52] = s2d.mean(axis=1) - thr
        feats[:, 53] = (s2d > thr).mean(axis=1)

    feats[:, 54:58] = res_feats

    # Spring-neap (vectorized)
    sn_ph = _2PI * (future_epoch_sec - _SN_REF_SEC) / (SPRING_NEAP_PERIOD * 86400.0)
    feats[:, 58] = np.sin(sn_ph)
    feats[:, 59] = np.cos(sn_ph)

    # Decay (vectorized) -- uses corrected day_offsets
    feats[:, 60] = np.exp(-day_offsets / 7.0)
    if last_exc >= 0 and n_hist_hours > 0:
        gap_h = (n_hist_hours - 1 - last_exc) + day_offsets * 24
        feats[:, 61] = np.minimum(gap_h / (21.0 * 24.0), 1.0)
    else:
        feats[:, 61] = 1.0

    return feats


# ======================================================================
# G7v2 pipeline runner
# ======================================================================
def run_g7v2_pipeline(index, args, model_dir, is_perday):
    """Run full G7v2 pipeline. Returns array of per-window/per-day probabilities."""
    n_windows = len(index)

    if xgb is None:
        print("[G7v2] xgboost not available, returning 0.5")
        return np.full(n_windows, 0.5, dtype=np.float64)

    booster_path = model_dir / "booster.json"
    scaler_path = model_dir / "scaler_stats.npz"
    meta_path = model_dir / "inference_meta.json"
    if not booster_path.exists():
        print("[G7v2] booster.json not found, returning 0.5")
        return np.full(n_windows, 0.5, dtype=np.float64)

    t0 = _time.time()
    booster = xgb.Booster()
    booster.load_model(str(booster_path))
    print(f"[G7v2] Booster loaded ({_time.time()-t0:.1f}s)")

    sc = np.load(str(scaler_path))
    sc_mean = sc["mean"].astype(np.float32)
    sc_scale = sc["scale"].astype(np.float32)
    sc_mean = np.where(np.isfinite(sc_mean), sc_mean, 0.0).astype(np.float32)
    sc_scale = np.where(np.isfinite(sc_scale) & (np.abs(sc_scale) > 1e-12), sc_scale, 1.0).astype(np.float32)

    threshold_map = dict(OFFICIAL_THRESHOLDS)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "official_threshold_map" in meta:
            threshold_map.update(meta["official_threshold_map"])

    # Load test data & build numpy station tables
    t1 = _time.time()
    test_df = pd.read_csv(args.test_hourly, usecols=["time", "station_name", "sea_level"])
    test_df["time"] = pd.to_datetime(test_df["time"])
    test_df["date"] = test_df["time"].dt.floor("D")
    print(f"[G7v2] Test loaded ({_time.time()-t1:.1f}s, {len(test_df):,} rows)")

    t1 = _time.time()
    station_data = {}
    for stn, g in test_df.groupby("station_name"):
        stn = str(stn)
        g = g.sort_values("time")

        daily = g.groupby("date").agg(
            sl_mean=("sea_level", "mean"),
            sl_max=("sea_level", "max"),
            sl_min=("sea_level", "min"),
            sl_std=("sea_level", "std"),
        ).reset_index().sort_values("date")
        daily["sl_std"] = daily["sl_std"].fillna(0.0)
        daily["r3m"] = daily["sl_mean"].rolling(3, min_periods=1).mean()
        daily["r7m"] = daily["sl_mean"].rolling(7, min_periods=1).mean()
        daily["r3s"] = daily["sl_mean"].rolling(3, min_periods=1).std().fillna(0.0)
        daily["r7s"] = daily["sl_mean"].rolling(7, min_periods=1).std().fillna(0.0)

        dates_np = daily["date"].to_numpy()
        stn_mean = float(g["sea_level"].mean())
        stn_std = float(g["sea_level"].std())
        if np.isnan(stn_std) or stn_std < 1e-6:
            stn_std = 1.0

        hourly_by_date = {}
        for date_val, dg in g.groupby("date"):
            sl_arr = dg["sea_level"].dropna().to_numpy(dtype=np.float64)
            hourly_by_date[np.datetime64(date_val)] = sl_arr

        date_to_idx = {d: i for i, d in enumerate(dates_np)}

        station_data[stn] = {
            "dates": dates_np,
            "date_to_idx": date_to_idx,
            "sl_mean": daily["sl_mean"].to_numpy(dtype=np.float64),
            "sl_max": daily["sl_max"].to_numpy(dtype=np.float64),
            "sl_min": daily["sl_min"].to_numpy(dtype=np.float64),
            "sl_std": daily["sl_std"].to_numpy(dtype=np.float64),
            "r3m": daily["r3m"].to_numpy(dtype=np.float64),
            "r7m": daily["r7m"].to_numpy(dtype=np.float64),
            "r3s": daily["r3s"].to_numpy(dtype=np.float64),
            "r7s": daily["r7s"].to_numpy(dtype=np.float64),
            "stn_mean": stn_mean,
            "stn_std": stn_std,
            "hourly_by_date": hourly_by_date,
        }

        if stn not in threshold_map:
            threshold_map[stn] = stn_mean + 1.5 * stn_std

    del test_df
    print(f"[G7v2] Station tables built ({_time.time()-t1:.1f}s, {len(station_data)} stations)")

    # Load train stats for stn_mean/stn_std override
    try:
        t1 = _time.time()
        tr = pd.read_csv(args.train_hourly, usecols=["station_name", "sea_level"])
        for stn_name, sg in tr.groupby("station_name"):
            stn_name = str(stn_name)
            if stn_name in station_data:
                m = float(sg["sea_level"].mean())
                s = float(sg["sea_level"].std())
                if not np.isnan(s) and s > 1e-6:
                    station_data[stn_name]["stn_mean"] = m
                    station_data[stn_name]["stn_std"] = s
        del tr
        print(f"[G7v2] Train stats loaded ({_time.time()-t1:.1f}s)")
    except Exception as e:
        print(f"[G7v2] Skipping train stats: {e}")

    # Pre-extract index columns as numpy
    hist_ends_np = index["hist_end"].to_numpy()
    fut_starts_np = index["future_start"].to_numpy()
    fut_ends_np = index["future_end"].to_numpy()
    stn_names = index["station_name"].to_numpy().astype(str)

    one_day = np.timedelta64(1, "D")

    # Build ALL features
    t1 = _time.time()
    all_feats = []
    window_ndays = []
    ok_mask = np.zeros(n_windows, dtype=bool)

    for wi in range(n_windows):
        station = stn_names[wi]
        if station not in station_data:
            continue

        obj = station_data[station]
        thr = threshold_map.get(station, 2.8)
        d2i = obj["date_to_idx"]
        hbd = obj["hourly_by_date"]

        hist_end = hist_ends_np[wi]
        hist_start = hist_end - np.timedelta64(HIST_DAYS - 1, "D")

        hist_indices = []
        d = hist_start
        while d <= hist_end:
            if d in d2i:
                hist_indices.append(d2i[d])
            d += one_day

        if len(hist_indices) < 3:
            continue

        while len(hist_indices) < HIST_DAYS:
            hist_indices.insert(0, hist_indices[0])
        hist_indices = hist_indices[-HIST_DAYS:]

        idx_arr = np.array(hist_indices)
        h_mean = obj["sl_mean"][idx_arr].astype(np.float32)
        h_max = obj["sl_max"][idx_arr].astype(np.float32)
        h_min = obj["sl_min"][idx_arr].astype(np.float32)
        h_std = obj["sl_std"][idx_arr].astype(np.float32)
        last_idx = hist_indices[-1]
        r3m = float(obj["r3m"][last_idx])
        r7m = float(obj["r7m"][last_idx])
        r3s = float(obj["r3s"][last_idx])
        r7s = float(obj["r7s"][last_idx])

        hourly_chunks = []
        d = hist_start
        while d <= hist_end:
            if d in hbd:
                hourly_chunks.append(hbd[d])
            d += one_day
        if hourly_chunks:
            hourly_sl = np.concatenate(hourly_chunks)
        else:
            hourly_sl = np.array([], dtype=np.float64)

        harmonics, tidal_mean, n_hh, res_f, last_exc = compute_tidal_info(hourly_sl, thr)

        fut_s = fut_starts_np[wi]
        fut_e = fut_ends_np[wi]
        n_fd = int((fut_e - fut_s) / one_day) + 1
        n_fd = min(max(n_fd, 0), FUTURE_DAYS)
        if n_fd == 0:
            continue

        # CORRECTED: Calculate actual day offset from hist_end
        base_day_offset = max(1, int((fut_s - hist_end) / one_day))

        future_dates = fut_s + np.arange(n_fd) * one_day
        future_doy = np.array([
            (fd - fd.astype("datetime64[Y]")).astype(int) + 1
            for fd in future_dates
        ], dtype=np.float32)
        future_epoch = future_dates.astype("datetime64[s]").astype(np.float64)

        wf = build_window_features(
            h_mean, h_max, h_min, h_std,
            r3m, r7m, r3s, r7s,
            hourly_sl, thr, obj["stn_mean"], obj["stn_std"],
            n_fd, future_doy, future_epoch,
            harmonics, tidal_mean, n_hh,
            res_f, last_exc,
            base_day_offset=base_day_offset,
        )
        all_feats.append(wf)
        window_ndays.append(n_fd)
        ok_mask[wi] = True

    print(f"[G7v2] Features built ({_time.time()-t1:.1f}s, {sum(ok_mask)}/{n_windows} ok)")

    # Batch predict
    t1 = _time.time()
    predictions = np.full(n_windows, 0.5, dtype=np.float64)

    if all_feats:
        X = np.vstack(all_feats)
        del all_feats
        X = ((X - sc_mean) / sc_scale).astype(np.float32)
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)

        dm = xgb.DMatrix(X)
        del X
        probs = booster.predict(dm)
        del dm
        print(f"[G7v2] Predicted ({_time.time()-t1:.1f}s, {len(probs):,} samples)")

        # Aggregate to window level
        offset = 0
        ok_indices = np.where(ok_mask)[0]
        for i, wi in enumerate(ok_indices):
            nd = window_ndays[i]
            dp = probs[offset:offset + nd].astype(np.float64)
            dp = np.clip(dp, 1e-9, 1.0 - 1e-9)

            if is_perday:
                # Per-day mode: each entry is a single day, output raw prob
                predictions[wi] = float(dp.max())
            else:
                # Per-window mode: aggregate with complementary probability
                predictions[wi] = 1.0 - np.prod(1.0 - dp)

            offset += nd

    return predictions


# ======================================================================
# d2s2 pipeline runner
# ======================================================================
def run_d2s2_pipeline(index, args, model_dir):
    """Run full d2s2 pipeline. Returns (risk_scores, y_prob) arrays."""
    n = len(index)

    bundle_paths = sorted(_glob.glob(str(model_dir / "bundle_*.pkl")))
    if not bundle_paths:
        print("[d2s2] No bundle_*.pkl found")
        return np.full(n, 0.5, dtype=np.float32), np.full(n, 0.5, dtype=np.float32)

    bundles = []
    d2s2_threshold_map = {}
    for bp in bundle_paths:
        with open(bp, "rb") as f:
            b = pickle.load(f)
        if str(b.get("model_kind", "")) != "f1push_ranker_v1":
            continue
        bundles.append(b)
        if not d2s2_threshold_map:
            d2s2_threshold_map = dict(b.get("official_threshold_map", {}))
    if not bundles:
        print("[d2s2] No valid f1push bundles loaded")
        return np.full(n, 0.5, dtype=np.float32), np.full(n, 0.5, dtype=np.float32)

    print(f"[d2s2] Loaded {len(bundles)} bundles")

    # Prepare index with keys
    idx = index.copy()
    idx["hist_end_str"] = idx["hist_end"].dt.strftime("%Y-%m-%d")
    idx["key"] = idx["station_name"].astype(str) + "|" + idx["hist_end_str"]

    test_hourly = pd.read_csv(args.test_hourly)
    daily = daily_aggregate(test_hourly)
    daily = attach_relative_features(daily, threshold_map=d2s2_threshold_map)

    prob_rows = []
    for b in bundles:
        prob_rows.append(score_bundle(b, daily=daily, index_df=idx))
    prob_matrix = np.asarray(prob_rows, dtype=np.float32)
    risk = consensus_risk(prob_matrix=prob_matrix)

    # Build d2s2 probabilities
    order = np.argsort(risk)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    rank_norm = rank / max(n - 1, 1)
    y_prob = (HIGH_PROB_VALUE - 0.02 * (1.0 - rank_norm)).astype(np.float32)

    low_idx = station_quota_selection(
        index_df=idx,
        risk_score=risk,
        q=float(FORCE_LOWEST_Q_TO_NEG),
        station_neg_prior=STATION_NEG_PRIOR,
        alpha=float(NEG_PRIOR_ALPHA),
    )
    if len(low_idx) > 0:
        y_prob[low_idx] = LOW_PROB_VALUE

    return risk, y_prob


def write_default(index_df, out_path, prob=0.5):
    pd.DataFrame({"id": index_df["id"], "y_prob": prob}).to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", required=True)
    ap.add_argument("--test_hourly", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--predictions_out", required=True)
    args = ap.parse_args()

    model_dir = Path(__file__).resolve().parent
    index = pd.read_csv(args.test_index)
    for col in ["hist_start", "hist_end", "future_start", "future_end"]:
        if col in index.columns:
            index[col] = pd.to_datetime(index[col])
    if "station_name" not in index.columns:
        index["station_name"] = "unknown"
    n_windows = len(index)

    # Detect mode: per-day vs per-window
    future_spans = (index["future_end"] - index["future_start"]).dt.days
    median_span = float(future_spans.median())
    is_perday = median_span < 7
    print(f"[Hybrid-Final] Detected mode: {'per-day' if is_perday else 'per-window'} "
          f"(median_span={median_span}, n_windows={n_windows})")

    try:
        ids = index["id"].to_numpy()

        if is_perday:
            # ── Per-day mode: G7v2 only ──
            print("[Hybrid-Final] Running G7v2-only pipeline for per-day mode")
            g7_probs = run_g7v2_pipeline(index, args, model_dir, is_perday=True)
            y_prob = np.clip(g7_probs, 0.0, 1.0)

        else:
            # ── Per-window mode: d2s2 + G7v2 rank ensemble ──
            print("[Hybrid-Final] Running d2s2 + G7v2 ensemble for per-window mode")

            # Run d2s2
            t0 = _time.time()
            d2s2_risk, d2s2_yprob = run_d2s2_pipeline(index, args, model_dir)
            print(f"[Hybrid-Final] d2s2 done ({_time.time()-t0:.1f}s)")

            # Run G7v2
            t0 = _time.time()
            g7_probs = run_g7v2_pipeline(index, args, model_dir, is_perday=False)
            print(f"[Hybrid-Final] G7v2 done ({_time.time()-t0:.1f}s)")

            n = len(index)

            # Rank d2s2 risk scores
            d2s2_order = np.argsort(d2s2_risk)
            d2s2_rank = np.empty(n, dtype=np.float64)
            d2s2_rank[d2s2_order] = np.arange(n, dtype=np.float64) / max(n - 1, 1)

            # Rank G7v2 probabilities
            g7_order = np.argsort(g7_probs)
            g7_rank = np.empty(n, dtype=np.float64)
            g7_rank[g7_order] = np.arange(n, dtype=np.float64) / max(n - 1, 1)

            # Rank-combine: 0.6 * d2s2_rank + 0.4 * g7v2_rank
            combined_rank = 0.6 * d2s2_rank + 0.4 * g7_rank

            # Map combined ranking to d2s2-style probabilities
            final_order = np.argsort(combined_rank)
            final_rank = np.empty(n, dtype=np.int64)
            final_rank[final_order] = np.arange(n)
            final_rank_norm = final_rank.astype(np.float64) / max(n - 1, 1)
            y_prob = (HIGH_PROB_VALUE - 0.02 * (1.0 - final_rank_norm)).astype(np.float64)

            # Apply station quota for forced-low
            # Use d2s2 risk for quota selection (it's the primary model)
            idx_for_quota = index.copy()
            low_idx = station_quota_selection(
                index_df=idx_for_quota,
                risk_score=combined_rank,  # Use combined rank for selection
                q=float(FORCE_LOWEST_Q_TO_NEG),
                station_neg_prior=STATION_NEG_PRIOR,
                alpha=float(NEG_PRIOR_ALPHA),
            )
            if len(low_idx) > 0:
                y_prob[low_idx] = LOW_PROB_VALUE

            print(f"[Hybrid-Final] Ensemble weights: d2s2=0.6, g7v2=0.4")
            print(f"[Hybrid-Final] d2s2 mean_rank={d2s2_rank.mean():.3f} "
                  f"g7v2 mean_rank={g7_rank.mean():.3f}")

        out = pd.DataFrame({"id": ids, "y_prob": np.clip(y_prob, 0.0, 1.0)})
        out.to_csv(args.predictions_out, index=False)

        pb = (y_prob >= 0.5).astype(int)
        print(f"[Hybrid-Final] Done: {len(out)} predictions, "
              f"mean={y_prob.mean():.4f} flood_rate={pb.mean():.4f}")

    except Exception:
        traceback.print_exc()
        write_default(index, args.predictions_out)


if __name__ == "__main__":
    main()
