"""Scoring harness: derive macro targets from a weekly curve, and compute metrics.

Weekly-curve metrics and macro-scalar metrics let us compare the weekly-series models against
the season-level models on common ground (identical labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def derive_macro(curve: pd.Series) -> dict:
    """Derive the three macroscopic targets from a predicted weekly incidence curve."""
    inc = curve.to_numpy(float)
    weeks = np.asarray(curve.index)
    if np.all(np.isnan(inc)) or np.nansum(inc) == 0:
        return dict(size_peak_incidence=0.0, size_attack_rate=0.0, peak_timing_week=np.nan)
    pos = int(np.nanargmax(inc))
    return dict(
        size_peak_incidence=float(inc[pos]),
        size_attack_rate=float(np.nansum(inc)),
        peak_timing_week=float(weeks[pos]),
    )


def weekly_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if mask.sum() == 0:
        return dict(MAE=np.nan, RMSE=np.nan)
    e = pred[mask] - obs[mask]
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))))


def derive_macro_paths(paths: np.ndarray, weeks: np.ndarray) -> dict:
    """Vectorized macro-target derivation across simulated paths. paths: (n_sims, W)."""
    inc = np.asarray(paths, float)
    weeks = np.asarray(weeks, float)
    tot = np.nansum(inc, axis=1)
    zero = tot <= 0
    peak = np.where(zero, 0.0, np.nanmax(inc, axis=1))
    timing = np.where(zero, np.nan, weeks[np.nanargmax(inc, axis=1)])
    return {"size_peak_incidence": peak, "size_attack_rate": tot, "peak_timing_week": timing}


def crps_ensemble(samples: np.ndarray, y: float) -> float:
    """Fair (sort-based) CRPS estimator for an ensemble forecast vs scalar observation."""
    s = np.sort(np.asarray(samples, float))
    s = s[~np.isnan(s)]
    n = len(s)
    if n == 0 or not np.isfinite(y):
        return np.nan
    t1 = np.mean(np.abs(s - y))
    i = np.arange(1, n + 1)
    t2 = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * s)   # = E|X - X'|
    return float(t1 - 0.5 * t2)


def probabilistic_metrics(samples_per_unit: list, obs: list) -> dict:
    """CRPS + interval coverage/width across units. samples_per_unit may contain None."""
    crps, cov50, cov90, w90 = [], [], [], []
    for s, y in zip(samples_per_unit, obs):
        if s is None or not np.isfinite(y):
            continue
        s = np.asarray(s, float)
        s = s[~np.isnan(s)]
        if len(s) == 0:
            continue
        crps.append(crps_ensemble(s, y))
        q05, q25, q75, q95 = np.percentile(s, [5, 25, 75, 95])
        cov50.append(q25 <= y <= q75)
        cov90.append(q05 <= y <= q95)
        w90.append(q95 - q05)
    if not crps:
        return {}
    return {"CRPS": float(np.mean(crps)), "coverage_50": float(np.mean(cov50)),
            "coverage_90": float(np.mean(cov90)), "width_90": float(np.mean(w90))}


def quantile_metrics(y_true, q_pred, quantiles) -> dict:
    """Scoring for predictive *quantiles* (vs samples): pinball loss + 90% interval cov/width.

    q_pred: (n, n_quantiles) on the natural scale, columns ordered as `quantiles`.
    """
    y = np.asarray(y_true, float)
    q = np.asarray(q_pred, float)
    mask = ~np.isnan(y) & ~np.isnan(q).any(axis=1)
    y, q = y[mask], q[mask]
    if len(y) == 0:
        return {}
    pin = []
    for j, a in enumerate(quantiles):
        e = y - q[:, j]
        pin.append(np.mean(np.maximum(a * e, (a - 1) * e)))
    out = {"pinball": float(np.mean(pin))}
    qs = [round(float(a), 4) for a in quantiles]
    if 0.05 in qs and 0.95 in qs:
        lo, hi = q[:, qs.index(0.05)], q[:, qs.index(0.95)]
        out["coverage_90"] = float(np.mean((y >= lo) & (y <= hi)))
        out["width_90"] = float(np.mean(hi - lo))
    return out


def macro_metrics(obs: np.ndarray, pred: np.ndarray, target: str) -> dict:
    """Cross-unit metrics for one target; metric set depends on the target type."""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    o, p = obs[mask], pred[mask]
    if len(o) == 0:
        return {}
    e = p - o
    out = dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))))
    if target in ("size_peak_incidence", "size_attack_rate"):
        denom = np.abs(o) + np.abs(p)
        out["sMAPE"] = float(np.mean(np.where(denom > 0, 2 * np.abs(e) / denom, 0.0)) * 100)
    if target in ("duration_mem_weeks", "peak_timing_week"):
        out["within_1wk"] = float(np.mean(np.abs(e) <= 1))
        out["within_2wk"] = float(np.mean(np.abs(e) <= 2))
    return out
