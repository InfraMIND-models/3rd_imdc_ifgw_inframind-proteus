"""Target construction. Builds the three macroscopic labels per (unit, season):
`size_peak_incidence`, `size_attack_rate`, and `peak_timing_week`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import DataRepository


def build_labels(repo: DataRepository) -> pd.DataFrame:
    """Return one row per labelable (unit, season) with the three targets.

    A season is labeled only if it is complete (>= MIN_SEASON_WEEKS) and has cases.
    """
    panel = repo.panel()
    unit_col = repo.unit_col
    curves = {key: g for key, g in panel.groupby([unit_col, "season"], sort=True)}

    # unit -> sorted seasons, and (unit, season) -> incidence array
    seasons_by_unit: dict[int, list[int]] = {}
    inc_by_key: dict[tuple, np.ndarray] = {}
    for (u, s), g in curves.items():
        seasons_by_unit.setdefault(u, []).append(s)
        inc_by_key[(u, s)] = g["incidence"].to_numpy(float)
    for u in seasons_by_unit:
        seasons_by_unit[u].sort()

    def labelable(u, s):
        g = curves[(u, s)]
        if len(g) < config.MIN_SEASON_WEEKS:
            return False
        inc = inc_by_key[(u, s)]
        return not (np.all(np.isnan(inc)) or np.nansum(g["casos"].to_numpy()) == 0)

    rows = []
    for u, seasons in seasons_by_unit.items():
        for s in seasons:
            if not labelable(u, s):
                continue
            g = curves[(u, s)]
            inc = inc_by_key[(u, s)]
            wis = g["week_in_season"].to_numpy()
            ews = g["epiweek"].to_numpy()
            peak_pos = int(np.nanargmax(inc))

            rows.append(dict(
                unit=u,
                season=s,
                n_weeks=len(g),
                size_peak_incidence=float(inc[peak_pos]),
                size_attack_rate=float(np.nansum(inc)),
                peak_timing_week=int(wis[peak_pos]),
                peak_epiweek=int(ews[peak_pos]),
            ))
    return pd.DataFrame(rows)
