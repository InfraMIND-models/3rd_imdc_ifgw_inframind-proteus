"""General modeling harness: build labels, run models over IMDC folds, score, save.

Usage:
    python3 -m inframind_proteus.outbreak_features.run <spatial_level> [models|default] [n_jobs]
e.g. ... run state                       # all models, serial
     ... run municipality default -1     # baselines+sarimax, all cores

`n_jobs != 1` parallelizes the per-unit work across units.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import config
from .baselines import Climatology, SeasonalNaive, observed_curve
from .data import DataRepository
from .evaluate import (derive_macro, derive_macro_paths, macro_metrics,
                       probabilistic_metrics, weekly_metrics)
from .labels import build_labels
from .sarimax import SarimaxModel

# name -> factory(repo, n_sims) -> model instance
REGISTRY = {
    "seasonal_naive": lambda repo, n_sims: SeasonalNaive(),
    "climatology": lambda repo, n_sims: Climatology(),
    "sarimax": lambda repo, n_sims: SarimaxModel(n_sims=n_sims),
}
_SLIM_COLS = ["epiweek", "season", "week_in_season", "incidence"]


def _row(model, level, fold, ts, target, metric, value, n):
    return dict(model=model, spatial_level=level, fold=fold, target_season=ts,
                target=target, metric=metric, value=value, n_units=n)


def _compute_unit(udf, meta, lab, models) -> dict:
    """Run every model on one unit; return per-model point/obs/weekly/sample results."""
    obs_curve = observed_curve(udf, meta["target_season"])
    wi = obs_curve.index
    res = {}
    for model in models:
        pc = model.predict_curve(udf, meta, wi)
        dm = derive_macro(pc)
        entry = dict(
            point={t: dm[t] for t in config.TARGETS},
            obs={t: float(lab[t]) for t in config.TARGETS},
            wo=obs_curve.to_numpy(float), wp=pc.to_numpy(float),
            fellback=bool(getattr(model, "last_fellback", False)), samp=None,
        )
        if getattr(model, "supports_uncertainty", False) and getattr(model, "last_paths", None) is not None:
            entry["samp"] = derive_macro_paths(model.last_paths, np.asarray(wi, float))
        res[model.name] = entry
    return res


def _compute_unit_fresh(udf, meta, lab, model_names, n_sims) -> dict:
    """Parallel worker: build fresh (no-exog) models per unit, then compute."""
    models = [REGISTRY[name](None, n_sims) for name in model_names]
    return _compute_unit(udf, meta, lab, models)


def run(spatial_level="state", model_names=None, n_jobs=1, n_sims=None) -> pd.DataFrame:
    n_sims = config.SARIMAX_N_SIMS if n_sims is None else n_sims
    if model_names is None:
        model_names = list(REGISTRY)

    repo = DataRepository(spatial_level)
    panel = repo.panel()
    labels = build_labels(repo)
    label_idx = labels.set_index(["unit", "season"])
    folds = repo.folds()
    cols = [repo.unit_col] + _SLIM_COLS
    unit_groups = {u: g.sort_values("epiweek")[cols] for u, g in panel.groupby(repo.unit_col)}
    serial_models = ([REGISTRY[n](repo, n_sims) for n in model_names] if n_jobs == 1 else None)

    rows, preds = [], []
    for k, meta in folds.items():
        ts = meta["target_season"]
        units = labels.loc[labels["season"] == ts, "unit"].unique()
        if len(units) == 0:
            print(f"[fold {k}] season {ts}: no complete labels -> skipped")
            continue
        print(f"[fold {k}] season {ts}: {len(units)} units x {len(model_names)} models "
              f"(n_jobs={n_jobs})")
        t_start = time.time()

        payloads = [(unit_groups[u], meta, label_idx.loc[(u, ts)].to_dict()) for u in units]
        if n_jobs == 1:
            outs = [_compute_unit(udf, m, lab, serial_models) for udf, m, lab in payloads]
        else:
            from joblib import Parallel, delayed
            outs = Parallel(n_jobs=n_jobs, batch_size=64)(
                delayed(_compute_unit_fresh)(udf, m, lab, model_names, n_sims) for udf, m, lab in payloads)

        fold_rows, fold_preds = _score_fold(outs, units, model_names, spatial_level, k, ts)
        rows += fold_rows
        preds.append(fold_preds)
        print(f"    fold {k} done in {time.time() - t_start:.1f}s")

    results = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"results_{spatial_level}.csv"
    results.to_csv(out, index=False)
    pd.concat(preds, ignore_index=True).to_csv(             # per-unit preds for common-set rescoring
        config.RESULTS_DIR / f"predictions_weekly_{spatial_level}.csv", index=False)
    labels.to_csv(config.RESULTS_DIR / f"labels_{spatial_level}.csv", index=False)
    print(f"\nsaved: {out}")
    return results


def _score_fold(outs, units, model_names, level, k, ts) -> tuple[list, list]:
    acc = {n: dict(pred={t: [] for t in config.TARGETS}, obs={t: [] for t in config.TARGETS},
                   samp={t: [] for t in config.TARGETS}, wo=[], wp=[], fb=0) for n in model_names}
    pred_rows = []                                          # per-unit macro predictions
    for unit, out in zip(units, outs):
        for name, e in out.items():
            a = acc[name]
            for t in config.TARGETS:
                a["pred"][t].append(e["point"][t])
                a["obs"][t].append(e["obs"][t])
                a["samp"][t].append(e["samp"][t] if e["samp"] is not None else None)
                pred_rows.append(dict(model=name, fold=k, target_season=ts, target=t,
                                      unit=unit, obs=e["obs"][t], pred=e["point"][t]))
            a["wo"].append(e["wo"])
            a["wp"].append(e["wp"])
            a["fb"] += int(e["fellback"])

    rows = []
    n_units = len(outs)
    for name in model_names:
        a = acc[name]
        for metric, val in weekly_metrics(np.concatenate(a["wo"]), np.concatenate(a["wp"])).items():
            rows.append(_row(name, level, k, ts, "weekly_curve", metric, val, n_units))
        for t in config.TARGETS:
            o, p = np.array(a["obs"][t], float), np.array(a["pred"][t], float)
            n = int((~np.isnan(o) & ~np.isnan(p)).sum())
            for metric, val in macro_metrics(o, p, t).items():
                rows.append(_row(name, level, k, ts, t, metric, val, n))
            samp = a["samp"][t]
            if any(s is not None for s in samp):
                n_s = sum(s is not None for s in samp)
                for metric, val in probabilistic_metrics(samp, a["obs"][t]).items():
                    rows.append(_row(name, level, k, ts, t, metric, val, n_s))
        rows.append(_row(name, level, k, ts, "_meta", "fallback_frac", a["fb"] / n_units, n_units))
        if a["fb"]:
            print(f"    {name}: {a['fb']}/{n_units} units fell back to baseline")
    return rows, pd.DataFrame(pred_rows)


def summarize(results: pd.DataFrame) -> None:
    """Print MAE per target/model averaged over folds (the headline comparison)."""
    mae = results[results["metric"] == "MAE"]
    pivot = mae.pivot_table(index="target", columns="model", values="value", aggfunc="mean")
    print("\n=== mean MAE across folds (lower is better) ===")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "state"
    names = None if len(sys.argv) <= 2 or sys.argv[2] == "default" else sys.argv[2].split(",")
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    summarize(run(level, names, n_jobs=jobs))
