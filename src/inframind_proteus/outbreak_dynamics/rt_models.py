"""Reproduction number R(t) trajectory models.

Each model implements the :class:`BaseRT` interface: given a DataFrame of
parameters (one row per simulation), produce a 2D numpy array of R(t) values.

Available models
----------------
LogisticRT
    Negative-logistic (inverted sigmoid) R(t) with a bounded active window.
    Parameters: rt_logist_roff, rt_logist_start, rt_logist_end,
                rt_logist_center, rt_logist_width, rt_logist_rmin,
                rt_logist_rmax
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRT(ABC):
    """Abstract base class for R(t) trajectory generators.

    Subclasses must implement :meth:`generate` and declare
    :attr:`required_params`.
    """

    #: Column names that must be present in the ``params_df`` passed to
    #: :meth:`generate`.  Subclasses should override this list.
    required_params: list[str] = []

    @abstractmethod
    def generate(
        self,
        params_df: pd.DataFrame,
        num_time_steps: int,
        step_dt: int,
        t_start: float = 0.0,
    ) -> np.ndarray:
        """Generate a 2D array of R(t) trajectories.

        Parameters
        ----------
        params_df:
            Parameter table.  One row per simulation; columns as declared
            in :attr:`required_params` (plus any optional extras).
        num_time_steps:
            Total number of time steps to produce (warm-up + simulation).
        step_dt:
            Duration of each time step in days.
        t_start:
            Day offset of the first time step, measured in days from
            ``zero_date``.  Defaults to 0.0.  Pass a negative value when
            the warm-up window starts before ``zero_date``.

        Returns
        -------
        np.ndarray
            Shape ``(num_simulations, num_time_steps)``.
        """
        ...

    def validate_params(self, params_df: pd.DataFrame) -> None:
        """Raise ``ValueError`` if any required parameter column is missing."""
        missing = [p for p in self.required_params if p not in params_df.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: missing required parameter columns: {missing}"
            )


class LogisticRT(BaseRT):
    """Negative-logistic (inverted sigmoid) R(t) with a bounded active window.

    The trajectory has three regions:

    * ``t < rt_logist_start``  : flat at ``rt_logist_roff``
    * ``rt_logist_start <= t < rt_logist_end`` : declining logistic curve
      starting near ``rt_logist_rmax`` and asymptoting to ``rt_logist_rmin``
    * ``t >= rt_logist_end``   : returns to ``rt_logist_roff``

    The logistic formula (decreasing from *rmax* toward *rmin*) is::

        R(t) = rmax - (rmax - rmin) / (1 + exp((center - t) / width))

    Expected columns in ``params_df``
    ----------------------------------
    rt_logist_roff    : R value outside the active window (off-season baseline)
    rt_logist_start   : Day (float) at which the logistic window opens
    rt_logist_end     : Day (float) at which the logistic window closes
    rt_logist_center  : Inflection point of the logistic (days)
    rt_logist_width   : Steepness parameter (days); larger = smoother decline
    rt_logist_rmin    : Asymptotic minimum R during outbreak
    rt_logist_rmax    : Peak / "R0"-like maximum R at outbreak onset
    """

    required_params: list[str] = [
        "rt_logist_roff",
        "rt_logist_start",
        "rt_logist_end",
        "rt_logist_center",
        "rt_logist_width",
        "rt_logist_rmin",
        "rt_logist_rmax",
    ]

    def generate(
        self,
        params_df: pd.DataFrame,
        num_time_steps: int,
        step_dt: int,
        t_start: float = 0.0,
    ) -> np.ndarray:
        """Generate logistic R(t) trajectories.

        Parameters
        ----------
        params_df:
            One row per simulation; must contain all :attr:`required_params`.
        num_time_steps:
            Number of time steps (columns in output).
        step_dt:
            Duration of each time step in days.
        t_start:
            Day offset of the first time step from ``zero_date``.

        Returns
        -------
        np.ndarray
            Shape ``(num_simulations, num_time_steps)``.
        """
        self.validate_params(params_df)

        # Time axis in days from zero_date: shape (num_time_steps,)
        time_grid = t_start + np.arange(num_time_steps, dtype=float) * step_dt

        # Per-simulation parameters, reshaped to (num_simulations, 1) for broadcasting
        rt_roff   = params_df["rt_logist_roff"].to_numpy()[:, np.newaxis]
        rt_start  = params_df["rt_logist_start"].to_numpy()[:, np.newaxis]
        rt_end    = params_df["rt_logist_end"].to_numpy()[:, np.newaxis]
        rt_center = params_df["rt_logist_center"].to_numpy()[:, np.newaxis]
        rt_width  = params_df["rt_logist_width"].to_numpy()[:, np.newaxis]
        rt_rmin   = params_df["rt_logist_rmin"].to_numpy()[:, np.newaxis]
        rt_rmax   = params_df["rt_logist_rmax"].to_numpy()[:, np.newaxis]

        # Active window: [start, end)  →  shape (num_simulations, num_time_steps)
        active = (time_grid >= rt_start) & (time_grid < rt_end)

        # Declining logistic: rmax at onset, rmin at saturation
        # exp() may overflow when the exponent is very large (t far before center),
        # which is correct — the logistic saturates to rmax in that limit.
        with np.errstate(over="ignore"):
            exponent = (rt_center - time_grid) / rt_width
            logistic_vals = rt_rmax - (rt_rmax - rt_rmin) / (1.0 + np.exp(exponent))

        # Outside the active window R(t) = roff
        return np.where(active, logistic_vals, rt_roff)
