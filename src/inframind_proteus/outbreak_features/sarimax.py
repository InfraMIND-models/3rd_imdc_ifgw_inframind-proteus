"""SARIMA + Fourier seasonality, per unit.

Models log1p(weekly incidence) per unit with ARIMA(p,d,q) + Fourier annual harmonics
(period ~52.18), avoiding a seasonal order s=52. Fits on weeks <= the fold issue point t0
(~EW25) and forecasts the full span up to the last target week (the IMDC leaves a ~15-week
gap between t0 and the EW41 season start, so the horizon is ~67 weeks); only the target
season's weeks are kept. Falls back to climatology on failure.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config
from .baselines import Climatology


def fourier_design(week_of_year, k: int, period: float = config.ANNUAL_PERIOD) -> np.ndarray:
    """Sin/cos harmonics for the annual cycle, phased by week-of-year (gap-robust)."""
    t = np.asarray(week_of_year, dtype=float)
    feats = []
    for h in range(1, k + 1):
        feats.append(np.sin(2 * np.pi * h * t / period))
        feats.append(np.cos(2 * np.pi * h * t / period))
    return np.column_stack(feats)


class SarimaxModel:
    def __init__(self, order=config.SARIMAX_ORDER, fourier_k=config.SARIMAX_FOURIER_K,
                 n_sims: int = 0, sim_cap_mult: float = config.SARIMAX_SIM_CAP_MULT):
        self.order = order
        self.fourier_k = fourier_k
        self.n_sims = n_sims
        self.sim_cap_mult = sim_cap_mult
        self.supports_uncertainty = n_sims > 0
        self.name = "sarimax"
        self._fallback = Climatology()
        self.n_fallback = 0
        self.n_fit = 0
        self.last_paths = None      # (n_sims, len(week_index)) set per call when simulating
        self.last_fellback = False  # whether the most recent call fell back to baseline

    def predict_curve(self, udf, fold_meta, week_index) -> pd.Series:
        self.last_paths = None
        self.last_fellback = False
        ts = fold_meta["target_season"]
        t0 = fold_meta["issue_epiweek"]

        train = udf[udf["epiweek"] <= t0].sort_values("epiweek")
        target = udf[udf["season"] == ts].sort_values("week_in_season")
        if len(target) == 0:
            return self._fb(udf, fold_meta, week_index)
        last_tgt = int(target["epiweek"].max())
        future = udf[(udf["epiweek"] > t0) & (udf["epiweek"] <= last_tgt)].sort_values("epiweek")

        y_inc = train["incidence"].to_numpy(float)
        ok = np.isfinite(y_inc)
        H = len(future)
        nonzero = int(np.sum(y_inc[ok] > 0))
        if (ok.sum() < config.SARIMAX_MIN_TRAIN_WEEKS or nonzero < config.SARIMAX_MIN_NONZERO_WEEKS
                or H == 0 or np.nansum(y_inc) <= 0):
            return self._fb(udf, fold_meta, week_index)  # sparsity / insufficient-history guard

        y = np.log1p(np.clip(y_inc[ok], 0, None))
        ew_train = train["epiweek"].to_numpy()[ok]
        ew_future = future["epiweek"].to_numpy()
        x_train = fourier_design(ew_train % 100, self.fourier_k)
        x_future = fourier_design(ew_future % 100, self.fourier_k)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(y, exog=x_train, order=self.order, trend="n",
                              enforce_stationarity=False, enforce_invertibility=False
                              ).fit(disp=False, maxiter=50, method="lbfgs")
                fc = np.asarray(res.get_forecast(steps=H, exog=x_future).predicted_mean, float)
            pred_all = np.expm1(fc)
        except Exception:
            return self._fb(udf, fold_meta, week_index)

        cap = 5.0 * float(np.nanmax(y_inc)) + 1.0
        if not np.all(np.isfinite(pred_all)) or np.nanmax(pred_all) > cap:
            return self._fb(udf, fold_meta, week_index)

        self.n_fit += 1
        # align forecast steps -> epiweeks, then keep the target season's weeks
        tgt_pos = pd.Index(ew_future).get_indexer(target["epiweek"].to_numpy())
        tgt_weeks = target["week_in_season"].to_numpy()
        curve = pd.Series(np.clip(pred_all, 0, None)[tgt_pos], index=tgt_weeks).reindex(week_index).fillna(0.0)

        if self.n_sims > 0:
            self._simulate(res, x_future, H, ew_future, target, week_index, float(np.nanmax(y_inc)))
        return curve

    def _simulate(self, res, x_future, H, ew_future, target, week_index, hist_max):
        """Draw n_sims forecast paths -> store (n_sims, len(week_index)) incidence array."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = res.simulate(nsimulations=H, anchor="end", exog=x_future, repetitions=self.n_sims)
            arr = np.asarray(sim, float).reshape(H, -1)[:, : self.n_sims]
            cap = self.sim_cap_mult * hist_max + 1.0
            paths = np.clip(np.expm1(arr), 0.0, cap)                       # (H, n_sims)
            pos = pd.Index(ew_future).get_indexer(target["epiweek"].to_numpy())
            ptgt = pd.DataFrame(paths[pos, :], index=target["week_in_season"].to_numpy())
            self.last_paths = ptgt.reindex(week_index).fillna(0.0).to_numpy().T  # (n_sims, W)
        except Exception:
            self.last_paths = None

    def _fb(self, udf, fold_meta, week_index) -> pd.Series:
        self.n_fallback += 1
        self.last_paths = None
        self.last_fellback = True
        return self._fallback.predict_curve(udf, fold_meta, week_index)
