"""The BaseModel contract for season-grain direct-regression models.

Every season-grain model subclasses this so the season harness (`run_season.py`) can treat
them uniformly. The weekly-curve forecasters (SARIMAX, baselines) keep their own surface in
`run.py` and are out of scope here.

Contract: a model receives an already-assembled design matrix `X` (one row per
(unit, season), indexed by (unit, season)) and a single target column `y`. It must not
reach back into raw data. `predict` returns predictions on the **natural** target scale
(any log1p train transform is inverted inside the model), so all models score identically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base"
    supports_uncertainty: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, *,
            cat_features: list[str] | None = None) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point prediction on the natural target scale, aligned to X's row order."""
        ...

    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles=(0.05, 0.5, 0.95)) -> pd.DataFrame:
        """Optional; required iff supports_uncertainty. Default raises."""
        raise NotImplementedError(f"{self.name} does not support uncertainty")

    def feature_importance(self, X: pd.DataFrame | None = None,
                           y: pd.Series | None = None) -> pd.DataFrame:
        """Driver attribution as tidy (feature, importance, [std]). Default raises."""
        raise NotImplementedError(f"{self.name} exposes no feature importance")

    def save(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> "BaseModel":
        raise NotImplementedError


def heldout_permutation_importance(model, X: pd.DataFrame, y, *, n_repeats: int,
                                   max_rows: int, seed: int = 0) -> pd.DataFrame:
    """Model-agnostic permutation importance: mean MAE rise (natural target scale) when each raw
    feature column of ``X`` is shuffled across rows, using only ``model.predict``.

    When ``(X, y)`` is the target-season fold the model never trained on, this is a genuinely
    **out-of-sample** driver basis. Each raw column is one feature, so a categorical is permuted
    as a unit automatically.

    Returns tidy (feature, importance, std), importance descending.
    """
    yv = np.asarray(y, float)
    keep = ~np.isnan(yv)
    X, yv = X.loc[keep], yv[keep]
    cols = list(X.columns)
    if len(X) < 3:                                  # too few held-out rows to permute meaningfully
        return pd.DataFrame({"feature": cols, "importance": np.nan, "std": np.nan})
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, yv = X.iloc[idx], yv[idx]
    n = len(X)
    base = float(np.mean(np.abs(np.asarray(model.predict(X), float) - yv)))
    rows = []
    for c in cols:
        orig = X[c].to_numpy(copy=True)
        diffs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = orig[rng.permutation(n)]
            diffs.append(float(np.mean(np.abs(np.asarray(model.predict(Xp), float) - yv))) - base)
        rows.append((c, float(np.mean(diffs)), float(np.std(diffs))))
    return pd.DataFrame(rows, columns=["feature", "importance", "std"]).sort_values(
        "importance", ascending=False, ignore_index=True)
