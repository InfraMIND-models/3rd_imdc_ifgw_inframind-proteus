"""Climatology-anomaly wrapper: fit/predict the *ratio to climatology*, not the raw target.

Climatology (each unit's mean of the target over strictly-prior seasons) is a strong
magnitude reference for the size targets. This wrapper hands any `BaseModel` that baseline
for free: it replaces the size-target `y` with the **multiplicative anomaly**
`y / clim(unit, season)` and post-multiplies the prediction by the same `clim`. The base
model then spends its capacity on the *deviation from the climatological norm* rather than
reconstructing each unit's endemic level.

Design choices:
  * **Multiplicative** (ratio), not additive: keeps the target positive so the base model's own
    `log1p` transform and `>=0` clipping are untouched -- the wrapper needs **no change** to any
    base model. Reconstruct as `pred_ratio * clim`.
  * **Rate targets only** (`config.LOG1P_TARGETS`). For `peak_timing_week` the base rounds
    predictions to integers, which is meaningless on a ~1.0 ratio, so it passes straight through
    to the base model unwrapped (the level-reconstruction rationale is a magnitude story anyway).
  * **Leakage-safe by construction**: `clim` for season s uses only training labels of seasons
    < s. At predict time every training season precedes the target season, so the offset *is*
    climatology and a zero-information base (ratio == 1) reproduces the climatology baseline exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .base_model import BaseModel
from .tabular import CatBoostModel


class BaselineOffsetModel(BaseModel):
    base_cls: type[BaseModel] = None      # set by subclasses

    def __init__(self, target: str, cat_features=None, repo=None, **kw):
        self.target = target
        self.apply = target in config.LOG1P_TARGETS      # only the rate (magnitude) targets
        self.base = self.base_cls(target, cat_features=cat_features, repo=repo, **kw)
        self.name = f"{self.base.name}_anom"
        self.supports_uncertainty = self.base.supports_uncertainty

    # ---- per-(unit, season) climatology from training labels (strictly prior seasons) ----
    def _build_offset(self, s: pd.Series) -> None:
        v = s.dropna()
        units = v.index.get_level_values("unit").to_numpy()
        seasons = v.index.get_level_values("season").to_numpy(float)
        vals = v.to_numpy(float)
        self._grand_ = float(np.mean(vals)) if len(vals) else 1.0
        self._unit_: dict[object, tuple[np.ndarray, np.ndarray]] = {}
        order = np.argsort(seasons, kind="stable")
        for u in np.unique(units):
            m = units[order] == u
            ss = seasons[order][m]
            self._unit_[u] = (ss, np.cumsum(vals[order][m]))   # one row per (unit, season) -> unique ss

    def _clim_for(self, index: pd.MultiIndex) -> np.ndarray:
        """Expanding-window unit mean over strictly-prior training seasons; grand mean if none."""
        out = np.empty(len(index), float)
        for i, (u, sq) in enumerate(zip(index.get_level_values("unit"),
                                        index.get_level_values("season"))):
            rec = self._unit_.get(u)
            if rec is None:
                out[i] = self._grand_
                continue
            ss, cs = rec
            k = int(np.searchsorted(ss, sq, side="left"))       # # of training seasons strictly < sq
            out[i] = cs[k - 1] / k if k > 0 else self._grand_
        out = np.where(out > 0, out, self._grand_)              # guard all-zero histories
        return np.clip(out, 1e-6, None)

    def fit(self, X: pd.DataFrame, y: pd.Series, *, cat_features=None) -> "BaselineOffsetModel":
        s = pd.Series(np.asarray(y, float), index=X.index)
        if not self.apply:                                      # non-rate target: passthrough
            self.base.fit(X, y, cat_features=cat_features)
        else:
            self._build_offset(s)
            ratio = s.to_numpy() / self._clim_for(X.index)
            self.base.fit(X, pd.Series(ratio, index=X.index), cat_features=cat_features)
        self.diagnostics_ = getattr(self.base, "diagnostics_", None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.apply:
            return self.base.predict(X)
        return np.clip(self.base.predict(X) * self._clim_for(X.index), 0.0, None)

    def predict_quantiles(self, X: pd.DataFrame, quantiles=None) -> pd.DataFrame:
        q = self.base.predict_quantiles(X, quantiles) if quantiles is not None \
            else self.base.predict_quantiles(X)
        if not self.apply:
            return q
        clim = pd.Series(self._clim_for(X.index), index=X.index)
        return q.mul(clim, axis=0)                              # quantiles scale multiplicatively

    def feature_importance(self, X=None, y=None) -> pd.DataFrame:
        return self.base.feature_importance(X, y)               # drivers explain the anomaly


class CatBoostAnomModel(BaselineOffsetModel):
    base_cls = CatBoostModel
