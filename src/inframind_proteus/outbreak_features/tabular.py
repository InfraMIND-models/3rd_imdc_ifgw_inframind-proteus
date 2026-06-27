"""CatBoost season-grain regressor (BaseModel) + season-grain baselines.

One model per target: log1p train transform for the rate targets, native categoricals +
native missing, early stopping on the latest train season (leakage-safe), native SHAP for
the driver report. Predictions are returned on the natural target scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .base_model import BaseModel


class CatBoostModel(BaseModel):
    supports_uncertainty = True    # per-quantile models -> predictive intervals

    def __init__(self, target: str, params: dict | None = None,
                 cat_features: list[str] | None = None,
                 quantiles=config.CATBOOST_QUANTILES, fit_quantiles: bool = True, repo=None):
        self.name = "catboost"
        self.target = target
        self.params = dict(params or config.CATBOOST_PARAMS)
        self.log_target = target in config.LOG1P_TARGETS
        self.round_target = target == "peak_timing_week"
        self._declared_cats = list(cat_features or [])
        self.quantiles = tuple(quantiles)
        self.fit_quantiles = fit_quantiles

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp = X.copy()
        for c in self.cat_features_:                       # CatBoost cats: str, no NaN
            Xp[c] = Xp[c].where(Xp[c].notna(), "NA").astype(str)
        return Xp[self.feature_names_]

    def _fit_one(self, Xp, yt, seasons, loss):
        """Fit one CatBoostRegressor, holding out the latest train season for early stopping."""
        from catboost import CatBoostRegressor, Pool
        model = CatBoostRegressor(**{**self.params, "loss_function": loss})
        uniq = np.unique(seasons)
        if len(uniq) >= 2:
            es = uniq.max()
            tr, va = seasons < es, seasons == es
            model.fit(Pool(Xp[tr], yt[tr], cat_features=self.cat_features_),
                      eval_set=Pool(Xp[va], yt[va], cat_features=self.cat_features_),
                      early_stopping_rounds=config.CATBOOST_EARLY_STOPPING, use_best_model=True)
        else:
            model.fit(Pool(Xp, yt, cat_features=self.cat_features_))
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series, *,
            cat_features: list[str] | None = None) -> "CatBoostModel":
        self.cat_features_ = [c for c in (cat_features or self._declared_cats) if c in X.columns]
        self.feature_names_ = list(X.columns)

        yv = pd.Series(y).to_numpy(float)
        keep = ~np.isnan(yv)
        Xp = self._prep(X).loc[keep]
        yt = np.log1p(yv[keep]) if self.log_target else yv[keep]
        seasons = X.index.get_level_values("season").to_numpy()[keep]

        self.model_ = self._fit_one(Xp, yt, seasons, self.params["loss_function"])  # point (RMSE)
        self.qmodels_ = ({q: self._fit_one(Xp, yt, seasons, f"Quantile:alpha={q}")
                          for q in self.quantiles} if self.fit_quantiles else {})
        return self

    def _to_natural(self, p: np.ndarray) -> np.ndarray:
        if self.log_target:
            p = np.expm1(p)
        p = np.clip(p, 0.0, None)
        if self.round_target:
            p = np.round(p)
        if self.target == "peak_timing_week":
            p = np.clip(p, 1.0, None)
        return p

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._to_natural(self.model_.predict(self._prep(X)))

    def predict_quantiles(self, X: pd.DataFrame, quantiles=None) -> pd.DataFrame:
        quantiles = tuple(quantiles or self.quantiles)
        if not self.qmodels_:
            raise NotImplementedError("model fitted with fit_quantiles=False")
        Xp = self._prep(X)
        arr = np.column_stack([self._to_natural(self.qmodels_[q].predict(Xp)) for q in quantiles])
        arr = np.sort(arr, axis=1)                         # enforce monotone (no quantile crossing)
        return pd.DataFrame(arr, index=X.index, columns=[float(q) for q in quantiles])

    def feature_importance(self, X: pd.DataFrame | None = None,
                           y: pd.Series | None = None) -> pd.DataFrame:
        from catboost import Pool
        pool = Pool(self._prep(X), cat_features=self.cat_features_)
        sv = self.model_.get_feature_importance(pool, type="ShapValues")[:, :-1]  # drop base value
        return pd.DataFrame({"feature": self.feature_names_,
                             "importance": np.abs(sv).mean(0),
                             "std": np.abs(sv).std(0)}).sort_values(
                                 "importance", ascending=False, ignore_index=True)

    def save(self, path) -> None:
        self.model_.save_model(str(path))


# ---- season-grain baselines (no new deps; for fair comparison) -------------
def _prior(labels: pd.DataFrame, target_season: int) -> pd.DataFrame:
    return labels[labels["season"] < target_season]


def season_naive(labels: pd.DataFrame, target: str, target_season: int,
                 units: np.ndarray) -> np.ndarray:
    """Each unit's most-recent prior-season value of the target."""
    prior = _prior(labels, target_season).sort_values("season")
    last = prior.groupby("unit")[target].last()
    return np.array([last.get(u, np.nan) for u in units], float)


def season_climatology(labels: pd.DataFrame, target: str, target_season: int,
                       units: np.ndarray) -> np.ndarray:
    """Each unit's mean of the target over all strictly-prior seasons."""
    mean = _prior(labels, target_season).groupby("unit")[target].mean()
    return np.array([mean.get(u, np.nan) for u in units], float)


SEASON_BASELINES = {"season_naive": season_naive, "season_climatology": season_climatology}
