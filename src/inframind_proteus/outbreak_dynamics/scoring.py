"""Scoring functions for evaluating simulation outputs against observations.

Primary method
--------------
``wis_score_vectorized`` — Weighted Interval Score (WIS) applied to
deterministic case beam quantiles.

Fallback metrics (individual trajectories)
------------------------------------------
``rmse_vectorized``, ``smape_vectorized`` — used when case beam scoring
is not applicable (e.g. single-trajectory calibration).

Helper
------
``nbinom_ppf_cf`` — fast Cornish-Fisher approximation to the Negative
Binomial PPF, used to build case beam quantiles without calling
``scipy.stats.nbinom.ppf`` for every simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Negative Binomial PPF (Cornish-Fisher approximation)
# ---------------------------------------------------------------------------

def nbinom_ppf_cf(
    q: float | np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    continuity: bool = True,
) -> np.ndarray:
    """Cornish-Fisher approximation to the Negative Binomial PPF.

    Fast vectorised alternative to ``scipy.stats.nbinom.ppf`` for computing
    case beam quantiles across many simulations simultaneously.

    Uses the SciPy parameterisation: X ~ nbinom(n, p) counts failures
    before n successes.

    Parameters
    ----------
    q:
        Quantile(s) in ``(0, 1)``.
    n:
        Number-of-successes parameter (> 0).
    p:
        Success-probability parameter in ``(0, 1)``.
    continuity:
        Apply a +0.5 continuity correction before returning.

    Returns
    -------
    np.ndarray
        Approximate quantile values (continuous, not rounded to integer).
    """
    # TODO: implement — port from proto_renewal_model.nbinom_ppf_cf
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Primary scoring: WIS over case beam quantiles
# ---------------------------------------------------------------------------

def wis_score_vectorized(
    simulations_df: pd.DataFrame,
    observations_sr: pd.Series,
    alphas: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    weight_of_median: float = 0.5,
) -> np.ndarray:
    """Compute the Weighted Interval Score (WIS) for multiple simulations.

    Scores deterministic case beam quantiles (prediction intervals) against
    observed case counts.

    Parameters
    ----------
    simulations_df:
        DataFrame with MultiIndex ``(quantile, i_simulation)`` and columns
        corresponding to time steps.  Must include the 0.5 (median) quantile.
    observations_sr:
        Observed case counts indexed by time step.  Must match
        ``simulations_df.columns``.
    alphas:
        Interval levels (e.g. ``[0.05, 0.5]`` for 95 % and 50 % PIs).
        Inferred from available quantiles when ``None``.
    weights:
        Per-interval weights.  Defaults to ``alphas / 2``.
    weight_of_median:
        Weight assigned to the median absolute error component.

    Returns
    -------
    np.ndarray
        Shape ``(num_simulations, num_time_steps)``.
    """
    # TODO: implement — port from proto_renewal_model.wis_score_vectorized
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Fallback scoring: individual trajectory metrics
# ---------------------------------------------------------------------------

def rmse_vectorized(
    simulations_df: pd.DataFrame,
    observations_sr: pd.Series,
) -> np.ndarray:
    """Root Mean Square Error for individual case trajectories.

    Parameters
    ----------
    simulations_df:
        DataFrame of shape ``(num_simulations, num_time_steps)``.
    observations_sr:
        Observed counts indexed by time step.

    Returns
    -------
    np.ndarray
        Shape ``(num_simulations,)``.
    """
    # TODO: implement
    raise NotImplementedError


def smape_vectorized(
    simulations_df: pd.DataFrame,
    observations_sr: pd.Series,
) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error for individual trajectories.

    Parameters
    ----------
    simulations_df:
        DataFrame of shape ``(num_simulations, num_time_steps)``.
    observations_sr:
        Observed counts indexed by time step.

    Returns
    -------
    np.ndarray
        Shape ``(num_simulations,)``.
    """
    # TODO: implement
    raise NotImplementedError
