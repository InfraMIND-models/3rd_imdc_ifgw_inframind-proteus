"""Reproduction number R(t) trajectory models.

Each model implements the :class:`BaseRT` interface: given a DataFrame of
parameters (one row per simulation), produce a 2D numpy array of R(t) values.

Available models
----------------
LogisticRT
    Negative-logistic (inverted sigmoid) R(t) with off-season activation.
    Parameters: rt_logist_roff, rt_logist_start, rt_logist_center,
                rt_logist_width, rt_logist_rmin, rt_logist_rmax
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
    """Negative-logistic (inverted sigmoid) R(t) with off-season activation.

    Before ``rt_logist_start`` the trajectory is flat at ``rt_logist_roff``.
    After that it follows a declining logistic curve from ``rt_logist_rmax``
    toward ``rt_logist_rmin``, centred at ``rt_logist_center``.

    Expected columns in ``params_df``
    ----------------------------------
    rt_logist_roff    : R value before activation (off-season baseline)
    rt_logist_start   : Day (float) at which the logistic activates
    rt_logist_center  : Inflection point of the logistic (days)
    rt_logist_width   : Steepness parameter (days); larger = smoother decline
    rt_logist_rmin    : Post-outbreak minimum R
    rt_logist_rmax    : Peak / "R0"-like maximum R
    """

    required_params: list[str] = [
        "rt_logist_roff",
        "rt_logist_start",
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
    ) -> np.ndarray:
        """Generate logistic R(t) trajectories.

        Returns
        -------
        np.ndarray
            Shape ``(num_simulations, num_time_steps)``.
        """
        self.validate_params(params_df)
        # TODO: implement — port from proto_renewal_model.create_logistic_rt
        raise NotImplementedError
