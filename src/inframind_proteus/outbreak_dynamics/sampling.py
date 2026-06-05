"""Sampling helpers for the outbreak dynamics module.

This module groups parameter sampling utilities used to build simulation
ensembles. Additional sampling helpers should be added here.
"""

from __future__ import annotations

import pandas as pd
import scipy.stats
from numpy.random import Generator


def sample_lhs(
    lhs_param_ranges: dict[str, list[float]],
    num_simulations: int,
    rng: Generator,
) -> pd.DataFrame:
    """Draw a Latin Hypercube sample scaled to the given parameter ranges.

    Parameters
    ----------
    lhs_param_ranges:
        Mapping from parameter name to ``[lower_bound, upper_bound]``.
    num_simulations:
        Number of samples (rows in the output).
    rng:
        NumPy random generator instance.

    Returns
    -------
    pd.DataFrame
        Shape ``(num_simulations, len(lhs_param_ranges))``, columns are
        parameter names.
    """
    lhs_sampler = scipy.stats.qmc.LatinHypercube(d=len(lhs_param_ranges), rng=rng)
    lhs_samples = lhs_sampler.random(n=num_simulations)

    l_bounds = [v[0] for v in lhs_param_ranges.values()]
    u_bounds = [v[1] for v in lhs_param_ranges.values()]
    lhs_scaled = scipy.stats.qmc.scale(lhs_samples, l_bounds, u_bounds)

    return pd.DataFrame(lhs_scaled, columns=list(lhs_param_ranges.keys()))
