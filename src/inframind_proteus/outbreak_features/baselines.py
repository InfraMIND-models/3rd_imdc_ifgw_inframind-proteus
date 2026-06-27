"""Baseline weekly-curve forecasters that every model must beat.

Each model exposes `predict_curve(udf, fold_meta, week_index)` where `udf` is the panel
slice for one unit (all seasons, sorted by epiweek) and `fold_meta` carries `target_season`
and `issue_epiweek`. Macro targets are derived from the returned curve (see evaluate.py),
so baselines and real models are scored on identical labels.
"""
from __future__ import annotations

import pandas as pd


def observed_curve(udf: pd.DataFrame, season) -> pd.Series:
    """Observed incidence for one season of one unit, indexed by week_in_season."""
    g = udf[udf["season"] == season].sort_values("week_in_season")
    return g.set_index("week_in_season")["incidence"]


class SeasonalNaive:
    """Predict the previous season's curve, aligned by within-season week."""

    name = "seasonal_naive"

    def predict_curve(self, udf, fold_meta, week_index) -> pd.Series:
        prev = observed_curve(udf, fold_meta["target_season"] - 1)
        if prev.empty:
            return pd.Series(0.0, index=week_index)
        return prev.reindex(week_index).fillna(0.0)


class Climatology:
    """Predict the per-week mean incidence over all strictly-prior seasons (leakage-safe)."""

    name = "climatology"

    def predict_curve(self, udf, fold_meta, week_index) -> pd.Series:
        hist = udf[udf["season"] < fold_meta["target_season"]]
        if hist.empty:
            return pd.Series(0.0, index=week_index)
        clim = hist.groupby("week_in_season")["incidence"].mean()
        return clim.reindex(week_index).fillna(0.0)
