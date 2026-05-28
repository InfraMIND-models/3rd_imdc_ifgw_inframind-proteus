"""Shared utilities for the outbreak dynamics module.

Includes:
- Latin Hypercube Sampling (LHS) for parameter space exploration
- Timestamp parsing: ISO date string, float days, or YYYYWW epiweek integer
- RNG seed helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import Generator


# ---------------------------------------------------------------------------
# LHS sampling
# ---------------------------------------------------------------------------

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
    # TODO: implement — port from proto_renewal_model.sample_lhs
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def parse_timestamp(
    value: str | int | float,
    zero_date: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Normalise a timestamp to a :class:`pandas.Timestamp`.

    Accepted formats:

    - ISO date string: ``"2023-10-02"``
    - Float days since ``zero_date``: ``0.0``, ``7.0``, …
    - YYYYWW epiweek integer: ``202340`` (→ Monday of that epiweek)

    Parameters
    ----------
    value:
        Input timestamp in any of the accepted formats.
    zero_date:
        Reference date for float-based offsets.  Required when ``value``
        is a ``float``.

    Returns
    -------
    pd.Timestamp
    """
    # TODO: implement
    #   - str  → pd.Timestamp(value)
    #   - int with 6 digits (YYYYWW) → epiweek_to_date(value)
    #   - float → zero_date + pd.Timedelta(days=value)
    raise NotImplementedError


def epiweek_to_date(epiweek_int: int) -> pd.Timestamp:
    """Convert a YYYYWW integer to the Monday of that epiweek.

    Parameters
    ----------
    epiweek_int:
        Integer of the form ``YYYYWW``, e.g. ``202340`` = week 40 of 2023.
        Uses the CDC epiweek convention (week starts on Sunday).

    Returns
    -------
    pd.Timestamp
        Monday of the given epiweek.
    """
    # TODO: implement using the epiweeks library
    raise NotImplementedError
