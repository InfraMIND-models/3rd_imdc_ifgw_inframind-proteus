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
    rt_logist_center  : Inflection point of the logistic (days) after start
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
            # exponent = (rt_center - time_grid) / rt_width  # Old: Center relative to date_zero
            exponent = (rt_start + rt_center - time_grid) / rt_width  # New: Center relative to start of active window
            logistic_vals = rt_rmax - (rt_rmax - rt_rmin) / (1.0 + np.exp(exponent))

        # Outside the active window R(t) = roff
        return np.where(active, logistic_vals, rt_roff)


class EnvelopedLogisticRT(BaseRT):
    """Negative-logistic (inverted sigmoid) R(t) with a logistic envelope for the
     active season.

    Modifies the prototype "LogisticRT" by smoothing the transitions between
    outbreak season and off-season with a logistic envelope.

    The baseline off-season value is assumed to be 1 and is not a parameter.

    Defining `logistic` as:
            logistic(t, center, width) = 1 / (1 + exp((center - t) / width))

    The R(t) core logistic decay is given by:
        C(t) = rmax - (rmax - rmin) * logistic(t, start + center, width)

    And the envelope is built as:
        E(t) = logistic(t, start, width) * (1 - logistic(t, start + end, width))

    The final expresison for R(t) is:
        R(t) = 1 + E(t) * (C(t) - 1)

    Expected columns in ``params_df``
    ----------------------------------
    rt_logist_start   :
        Day (float) at which the active season starts, with respect to date_zero.
    rt_logist_dt_center :
        Days since start of active season at which the core logistic inflects.
    rt_logist_dt_end  :
        Days since start of active season at which the active season ends.
    rt_logist_w_center :
        Exp width of the core logistic inflection (days); larger = smoother decline.
    rt_logist_w_env :
        Exp width of the envelope transitions (days); larger = smoother transition.
    rt_logist_r_low     : Asymptotic minimum R of the core logistic.
    rt_logist_r_high    : Asymptotic maximum R of the core logistic.
    """

    required_params: list[str] = [
        "rt_logist_start",
        "rt_logist_dt_center",
        "rt_logist_dt_end",
        "rt_logist_w_center",
        "rt_logist_w_env",
        "rt_logist_r_low",
        "rt_logist_r_high",
    ]

    @staticmethod
    def logistic(t, center, width):
        """Logistic function with value 0.5 at t=center and steepness controlled by width."""
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp((center - t) / width))

    def generate(
        self,
        params_df: pd.DataFrame,
        num_time_steps: int,
        step_dt: int,
        t_start: float = 0.0,
    ) -> np.ndarray:
        """Generate enveloped logistic R(t) trajectories.

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
        rt_start = params_df["rt_logist_start"].to_numpy()[:, np.newaxis]
        rt_dt_center = params_df["rt_logist_dt_center"].to_numpy()[:, np.newaxis]
        rt_dt_end = params_df["rt_logist_dt_end"].to_numpy()[:, np.newaxis]
        rt_w_center = params_df["rt_logist_w_center"].to_numpy()[:, np.newaxis]
        rt_w_env = params_df["rt_logist_w_env"].to_numpy()[:, np.newaxis]
        rt_r_low = params_df["rt_logist_r_low"].to_numpy()[:, np.newaxis]
        rt_r_high = params_df["rt_logist_r_high"].to_numpy()[:, np.newaxis]

        # Core logistic: decreasing from r_high to r_low, centered at start + dt_center
        center_abs = rt_start + rt_dt_center
        core_logistic = self.logistic(time_grid, center_abs, rt_w_center)
        C_t = rt_r_high - (rt_r_high - rt_r_low) * core_logistic

        # Envelope: rises at start, falls at start + dt_end
        envelope_rise = self.logistic(time_grid, rt_start, rt_w_env)
        envelope_fall = 1.0 - self.logistic(time_grid, rt_start + rt_dt_end, rt_w_env)
        E_t = envelope_rise * envelope_fall

        # Final R(t) = 1 + E(t) * (C(t) - 1)
        rt_trajectory = 1.0 + E_t * (C_t - 1.0)

        return rt_trajectory


# REGISTRY/factory
# =========

def get_rt_model(model_name: str) -> BaseRT:
    """Factory function to retrieve an R(t) model by name."""
    model_name = model_name.lower()
    if model_name == "logistic":
        return LogisticRT()
    elif model_name == "enveloped_logistic":
        return EnvelopedLogisticRT()
    else:
        raise ValueError(f"Unsupported reproduction number model: {model_name}")
