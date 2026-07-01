"""Generation time (GT) distribution models.

The GT is expressed as a PMF over discrete time-step intervals.
A single GT trajectory is shared across all simulations in one simulator
call — GT parameters live on the model object, not in the per-simulation
parameter table.  This keeps memory usage proportional to
``num_time_steps × gt_max_steps``, independent of ``num_simulations``.

Output convention
-----------------
``get_pmf`` returns shape ``(num_time_steps, gt_max_steps)``.

- **Axis 0** — simulation time step at which the PMF applies.
  Time-invariant models broadcast a single row, so no extra memory is
  allocated for this axis.
- **Axis 1** — generation time lag in reversed order: index 0 carries the
  weight for the *largest* lag (oldest contributing infection), index -1
  carries the weight for the *smallest* lag (most recent).
  This aligns directly with the infection history window
  ``infec[i_step - gt_max_steps : i_step]`` used in the renewal-equation
  dot-product.

Available models
----------------
ConstantGammaGT
    Gamma-shaped PMF, constant over the entire simulation period.
    Constructor parameters: ``shape``, ``scale`` (in days).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.stats


class BaseGT(ABC):
    """Abstract base class for generation time distributions.

    Subclasses must implement :meth:`get_pmf`.  GT parameters are stored
    on the object at construction time; they are **not** taken from the
    per-simulation ``params_df``.
    """

    @abstractmethod
    def get_pmf(
        self,
        gt_max_steps: int,
        num_time_steps: int,
        step_dt: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """Return the (reversed) generation time PMF.

        Parameters
        ----------
        gt_max_steps:
            Look-back window length in time steps.
        num_time_steps:
            Number of simulation time steps (length of the time axis).
        step_dt:
            Duration of each time step in days.
        normalize:
            Whether the PMF should be re-normalized to fix truncation effects.

        Returns
        -------
        np.ndarray
            Shape ``(num_time_steps, gt_max_steps)``.
            Values are non-negative; each row sums to ≤ 1 (truncated at
            ``gt_max_steps``).  Axis 1 is in **reversed lag order**:
            index 0 = largest lag, index -1 = smallest lag.
        """
        ...


class ConstantGammaGT(BaseGT):
    """Gamma-shaped generation time PMF, constant over the simulation period.

    Parameters
    ----------
    shape:
        Shape parameter ``a`` of the Gamma distribution (> 0).
    scale:
        Scale parameter in days; mean = shape × scale (> 0).
    """

    def __init__(self, shape: float, scale: float) -> None:
        if shape <= 0:
            raise ValueError(f"shape must be > 0, got {shape}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.shape = float(shape)
        self.scale = float(scale)

    def get_pmf(
        self,
        gt_max_steps: int,
        num_time_steps: int,
        step_dt: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute the gamma GT PMF and broadcast over the time axis.

        The PMF is computed once (a single 1-D vector) and broadcast to
        ``(num_time_steps, gt_max_steps)`` without allocating extra memory.

        Returns
        -------
        np.ndarray
            Shape ``(num_time_steps, gt_max_steps)``.
        """
        # CDF evaluation grid: one extra boundary point for np.diff
        # gt_vals[j] = j * step_dt  →  PMF[j] = P(lag in [j*dt, (j+1)*dt])
        gt_vals = np.arange(0, (gt_max_steps + 1) * step_dt, step_dt)  # (gt_max_steps+1,)

        cdf = scipy.stats.gamma.cdf(gt_vals, a=self.shape, scale=self.scale)  # (gt_max_steps+1,)
        pmf = np.diff(cdf)
        # (gt_max_steps,)

        if normalize:
            pmf_sum = pmf.sum()
            if pmf_sum > 0:
                pmf /= pmf_sum
            else:
                raise ValueError(
                    f"GT PMF sum is zero, check shape={self.shape}, scale={self.scale}, "
                    f"gt_max_steps={gt_max_steps}, step_dt={step_dt}"
                )

        # Reverse: index 0 = largest lag, aligns with oldest entry in the
        # infection history window used by the renewal equation
        pmf_rev = pmf[::-1]                                                    # (gt_max_steps,)

        # Broadcast to (num_time_steps, gt_max_steps) — read-only view,
        # no extra memory allocated for the time axis
        return np.broadcast_to(pmf_rev[np.newaxis, :], (num_time_steps, gt_max_steps))

