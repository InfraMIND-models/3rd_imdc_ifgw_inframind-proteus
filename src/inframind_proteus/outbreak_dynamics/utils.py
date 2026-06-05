"""Shared utilities for the outbreak dynamics module.

Includes:
- Timestamp parsing: ISO date string, float days, or YYYYWW epiweek integer
- RNG seed helpers
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import pandas as pd


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def epiweek_to_date(epiweek_int: int) -> pd.Timestamp:
    """Convert a YYYYWW integer to the start date (Sunday) of that CDC epiweek.

    Parameters
    ----------
    epiweek_int:
        Integer of the form ``YYYYWW``, e.g. ``202340`` = week 40 of 2023.
        Uses the CDC epiweek convention (week starts on Sunday).

    Returns
    -------
    pd.Timestamp
        Sunday that opens the given CDC epiweek.
    """
    from epiweeks import Week

    year = epiweek_int // 100
    week = epiweek_int % 100
    return pd.Timestamp(Week(year, week, system="cdc").startdate())


def parse_timestamp(
    value: str | int | float,
    zero_date: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Normalise a timestamp to a :class:`pandas.Timestamp`.

    Accepted formats:

    - ISO date string: ``"2023-10-02"``
    - YYYYWW epiweek integer: ``202340`` (→ Sunday opening that CDC epiweek)
    - Float days since ``zero_date``: ``0.0``, ``7.0``, …

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
    if isinstance(value, str):
        return pd.Timestamp(value)
    if isinstance(value, int):
        return epiweek_to_date(value)
    if isinstance(value, float):
        if zero_date is None:
            raise ValueError(
                "zero_date must be provided when value is a float (days offset)."
            )
        return zero_date + pd.Timedelta(days=value)
    raise TypeError(
        f"Unsupported timestamp type: {type(value).__name__!r}. "
        "Expected str (ISO date), int (YYYYWW), or float (days offset)."
    )


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as fp:
        return yaml.safe_load(fp)
