"""Season-matrix harness: direct (unit, season) regression over the IMDC folds.

The tabular counterpart to `run.py` (which forecasts weekly curves). For each
(model x spatial_level x target x fold): assemble the season feature matrix (fit on train
seasons), fit one model per target, predict the target season, score with the shared macro
metrics, and extract driver importances. Writes the same tidy schema as `run.py` (no
weekly_curve rows).

Registered season models: `catboost`, `lstm`, and the opt-in climatology-anomaly wrapper
`catboost_anom`.

Usage:
    python3 -m inframind_proteus.outbreak_features.run_season <level> [models|default]
e.g. ... run_season state catboost_anom,lstm
     ... run_season state default        # catboost + season baselines
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config
from .base_model import heldout_permutation_importance
from .data import DataRepository
from .evaluate import macro_metrics, quantile_metrics
from .labels import build_labels
from .season_features import FeatureAssembler, season_t0
from .sequence import LSTMModel
from .residual import CatBoostAnomModel
from .tabular import SEASON_BASELINES, CatBoostModel

# name -> BaseModel class (the season models registered for this component).
MODEL_REGISTRY = {"catboost": CatBoostModel, "lstm": LSTMModel,
                  # climatology-anomaly wrapper (residual.py): fit the ratio to climatology.
                  # Opt-in only -- excluded from the `default` set below.
                  "catboost_anom": CatBoostAnomModel}
_OPT_IN = ("_anom",)
MODELS = [m for m in MODEL_REGISTRY if not m.endswith(_OPT_IN)] + list(SEASON_BASELINES)


def _row(model, level, fold, ts, target, metric, value, n):
    return dict(model=model, spatial_level=level, fold=fold, target_season=ts,
                target=target, metric=metric, value=value, n_units=n)


def run(level="state", model_names=None) -> pd.DataFrame:
    if model_names is None:
        model_names = list(MODELS)
    repo = DataRepository(level)
    labels = build_labels(repo)
    lab_idx = labels.set_index(["unit", "season"]).sort_index()
    asm = FeatureAssembler(repo, labels)
    folds = repo.folds()

    rows, importances, heldout, diags, preds = [], {}, {}, [], []
    for k, meta in folds.items():
        ts = meta["target_season"]
        # Leakage guard: the feature pipeline hardcodes t0 = EW25 of the start year
        # (season_features.season_t0), while data.py decodes the real issue point per fold from
        # the train mask. They must agree, or features would see weeks the IMDC withholds.
        issue = meta.get("issue_epiweek")
        if issue is not None:
            assert issue == season_t0(ts), (
                f"[fold {k}] decoded issue_epiweek {issue} != season_t0({ts})="
                f"{season_t0(ts)}; feature pipeline t0 and fold issue point disagree")
        train_seasons = [s for s in meta["train_seasons"] if s < ts]
        if not labels["season"].eq(ts).any():
            print(f"[fold {k}] season {ts}: no labels -> skipped")
            continue
        asm.fit(train_seasons)
        X_tr = asm.transform(train_seasons)
        X_ts = asm.transform([ts])
        units = X_ts.index.get_level_values("unit").to_numpy()
        print(f"[fold {k}] season {ts}: train {len(X_tr)} rows / {len(train_seasons)} seasons, "
              f"target {len(X_ts)} units, {X_tr.shape[1]} features")

        for target in config.TARGETS:
            y_true = lab_idx.reindex(X_ts.index)[target].to_numpy(float)
            for name in model_names:
                qpred = None
                if name in MODEL_REGISTRY:
                    y_tr = lab_idx.reindex(X_tr.index)[target]
                    model = MODEL_REGISTRY[name](target, cat_features=asm.cat_features, repo=repo)
                    model.fit(X_tr, y_tr, cat_features=asm.cat_features)
                    pred = model.predict(X_ts)
                    if model.supports_uncertainty:
                        qpred = model.predict_quantiles(X_ts)
                    fi = model.feature_importance(X_tr)
                    fi.insert(0, "fold", k); fi.insert(1, "target", target)
                    importances.setdefault(name, []).append(fi)
                    if config.HELDOUT_PERM and name in config.HELDOUT_PERM_MODELS:
                        ho = heldout_permutation_importance(   # out-of-sample driver basis
                            model, X_ts, y_true, n_repeats=config.HELDOUT_PERM_REPEATS,
                            max_rows=config.HELDOUT_PERM_MAX_ROWS)
                        ho.insert(0, "fold", k); ho.insert(1, "target", target)
                        heldout.setdefault(name, []).append(ho)
                    if getattr(model, "diagnostics_", None):
                        diags.append(dict(model=name, fold=k, target=target, **model.diagnostics_))
                else:
                    pred = SEASON_BASELINES[name](labels, target, ts, units)

                o, p = y_true, np.asarray(pred, float)
                n = int((~np.isnan(o) & ~np.isnan(p)).sum())
                preds.append(pd.DataFrame({"model": name, "fold": k, "target_season": ts,
                                           "target": target, "unit": units, "obs": o, "pred": p}))
                for metric, val in macro_metrics(o, p, target).items():
                    rows.append(_row(name, level, k, ts, target, metric, val, n))
                if qpred is not None:
                    for metric, val in quantile_metrics(o, qpred.to_numpy(),
                                                        config.CATBOOST_QUANTILES).items():
                        rows.append(_row(name, level, k, ts, target, metric, val, n))

    results = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"results_season_{level}.csv"
    if out.exists():                                  # merge: keep prior models, refresh the ones just run
        prior = pd.read_csv(out)
        prior = prior[~prior["model"].isin(results["model"].unique())]
        results_out = pd.concat([prior, results], ignore_index=True)
    else:
        results_out = results
    results_out.to_csv(out, index=False)

    pred_df = pd.concat(preds, ignore_index=True)          # per-unit preds for common-set rescoring
    pout = config.RESULTS_DIR / f"predictions_season_{level}.csv"
    if pout.exists():                                       # merge: keep prior models, refresh re-run ones
        prior = pd.read_csv(pout)
        prior = prior[~prior["model"].isin(pred_df["model"].unique())]
        pred_df = pd.concat([prior, pred_df], ignore_index=True)
    pred_df.to_csv(pout, index=False)

    for name, lst in importances.items():
        pd.concat(lst, ignore_index=True).to_csv(
            config.RESULTS_DIR / f"importances_{name}_{level}.csv", index=False)
    for name, lst in heldout.items():               # held-out (target-season) permutation basis
        pd.concat(lst, ignore_index=True).to_csv(
            config.RESULTS_DIR / f"importances_heldout_{name}_{level}.csv", index=False)
    if diags:
        d = pd.DataFrame(diags)
        for m, sub in d.groupby("model"):
            sub.dropna(axis=1, how="all").to_csv(
                config.RESULTS_DIR / f"diagnostics_{m}_{level}.csv", index=False)
    print(f"\nsaved: {out}")
    return results


def summarize(results: pd.DataFrame) -> None:
    """MAE per target/model averaged over folds (lower is better)."""
    mae = results[results["metric"] == "MAE"]
    pivot = mae.pivot_table(index="target", columns="model", values="value", aggfunc="mean")
    print("\n=== mean MAE across folds (lower is better) ===")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    lvl = sys.argv[1] if len(sys.argv) > 1 else "state"
    names = None if len(sys.argv) <= 2 or sys.argv[2] == "default" else sys.argv[2].split(",")
    summarize(run(lvl, names))
