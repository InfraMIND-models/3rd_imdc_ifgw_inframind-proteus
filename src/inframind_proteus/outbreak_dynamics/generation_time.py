"""Generation time (GT) distribution models.

The GT is expressed as a PMF over discrete weekly intervals.
Only a single trajectory of GT PMF arrays is used per simulator call
(time-dependent GT keeps the same object in memory).

Available models
----------------
ConstantGammaGT
    Gamma-shaped PMF, constant over the entire simulation period.
    Parameters: gt_gamma_shape, gt_gamma_scale
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseGT(ABC):
    """Abstract base class for generation time distributions.

    Subclasses must implement :meth:`get_pmf` and declare
    :attr:`required_params`.
    """

    #: Column names required in ``params_df`` by :meth:`get_pmf`.
    required_params: list[str] = []

    @abstractmethod
    def get_pmf(
        self,
        gt_max_steps: int,
        step_dt: int,
        params_df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute the (reversed) generation time PMF array.

        Parameters
        ----------
        gt_max_steps:
            Maximum number of GT steps (look-back window length).
        step_dt:
            Duration of each time step in days.
        params_df:
            Parameter table (one row per simulation).

        Returns
        -------
        np.ndarray
            Shape ``(num_simulations, gt_max_steps)``.
            Rows sum to ≤ 1 (truncated at ``gt_max_steps``).
            Returned in **reversed** order (index 0 = most recent lag)
            so it can be used directly in a dot-product with the infection
            history in the renewal equation.
        """
        ...

    def validate_params(self, params_df: pd.DataFrame) -> None:
        """Raise ``ValueError`` if any required parameter column is missing."""
        missing = [p for p in self.required_params if p not in params_df.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: missing required parameter columns: {missing}"
            )


class ConstantGammaGT(BaseGT):
    """Gamma-shaped generation time PMF, constant over the simulation period.

    Expected columns in ``params_df``
    ----------------------------------
    gt_gamma_shape : Shape parameter ``a`` of the Gamma distribution
    gt_gamma_scale : Scale parameter (mean = shape × scale, in days)
    """

    required_params: list[str] = ["gt_gamma_shape", "gt_gamma_scale"]

    def get_pmf(
        self,
        gt_max_steps: int,
        step_dt: int,
        params_df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute the gamma GT PMF, reversed for use in the renewal equation.

        Returns
        -------
        np.ndarray
            Shape ``(num_simulations, gt_max_steps)``.
        """
        self.validate_params(params_df)
        # TODO: implement — port from proto_renewal_model.ProtoDynModel.run_multiple
        #   1. Build time grid: np.arange(0, (gt_max_steps + 1) * step_dt, step_dt)
        #   2. Evaluate gamma CDF at each grid point (vectorised over simulations)
        #   3. Compute PMF via np.diff along the step axis
        #   4. Reverse along the step axis for convolution-ready output
        raise NotImplementedError
