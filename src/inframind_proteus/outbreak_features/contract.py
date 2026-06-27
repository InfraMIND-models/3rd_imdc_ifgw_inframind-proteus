"""Map the rate/week feature samples to the export-contract schema and write them out.

The models predict rates (/100k) and a within-season week index; the contract wants raw case
counts and a `peak_week` relative to EW40. The conversions (see export_contract.md):

    case_attack_rate = round(size_attack_rate  * population / 100_000)   # season total cases
    peak_amplitude   = round(size_peak_incidence * population / 100_000) # cases in the peak week
    peak_week        = peak_timing_week                                  # identical (1 = EW41)

Population is the UF population of the season **start year** (the same per-(unit, year) figure the
incidence rates were built from, so the conversion round-trips). Counts are rounded to
non-negative integers only at the very end (after sampling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import DataRepository

# our (target, scale) -> contract feature name
SIZE_MAP = {"size_attack_rate": "case_attack_rate", "size_peak_incidence": "peak_amplitude"}
TIMING_MAP = {"peak_timing_week": "peak_week"}
CONTRACT_FEATURES = ("case_attack_rate", "peak_amplitude", "peak_week")


def uf_acronyms() -> dict[int, str]:
    """uf_code (IBGE int) -> UF 2-letter acronym, from the demographic UF table."""
    t = pd.read_csv(config.REPO_ROOT / "data" / "demographic" / "uf_table.csv",
                    usecols=["uf", "uf_code"])
    return dict(zip(t["uf_code"].astype(int), t["uf"].astype(str)))


def start_year_population(repo: DataRepository) -> dict[tuple[int, int], float]:
    """(unit, year) -> UF population in that year (the season's start year)."""
    panel = repo.panel()
    pop = panel.groupby([repo.unit_col, "year"])["population"].first()
    return {(int(u), int(y)): float(p) for (u, y), p in pop.items()}


def to_contract(samples: pd.DataFrame, repo: DataRepository) -> pd.DataFrame:
    """Convert long rate/week samples to long contract samples.

    Input columns:  unit, year, target, i_sample, value   (rates /100k or week index)
    Output columns: location_id, year, i_sample, feature, value   (integer counts / week)
    """
    acro = uf_acronyms()
    pop = start_year_population(repo)
    missing_acro = sorted(set(samples["unit"]) - set(acro))
    if missing_acro:
        raise KeyError(f"no UF acronym for unit codes {missing_acro}")

    out = []
    for target, g in samples.groupby("target", sort=False):
        g = g.copy()
        if target in SIZE_MAP:
            feature = SIZE_MAP[target]
            popv = np.array([pop.get((int(u), int(y)), np.nan)
                             for u, y in zip(g["unit"], g["year"])])
            if np.isnan(popv).any():
                bad = sorted({(int(u), int(y)) for u, y, p in zip(g["unit"], g["year"], popv)
                              if np.isnan(p)})
                raise KeyError(f"no population for (unit, year) {bad}")
            val = np.clip(np.round(g["value"].to_numpy() * popv / config.INCIDENCE_SCALE), 0, None)
        elif target in TIMING_MAP:
            feature = TIMING_MAP[target]
            val = np.clip(np.round(g["value"].to_numpy()), 1, None)
        else:
            continue                                       # ignore any non-contract target
        out.append(pd.DataFrame({
            "location_id": [acro[int(u)] for u in g["unit"]],
            "year": g["year"].astype(int).to_numpy(),
            "i_sample": g["i_sample"].astype(int).to_numpy(),
            "feature": feature,
            "value": val.astype(np.int64),
        }))
    return pd.concat(out, ignore_index=True)


def write_contract(samples: pd.DataFrame, repo: DataRepository, out_dir=None) -> list:
    """Write one CSV per contract feature: <feature>.csv with columns
    [location_id, year, i_sample, <feature>]. Returns the written paths."""
    out_dir = config.REPO_ROOT / "predictions" if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = to_contract(samples, repo)
    paths = []
    for feature, g in contract.groupby("feature", sort=False):
        df = (g[["location_id", "year", "i_sample", "value"]]
              .rename(columns={"value": feature})
              .sort_values(["location_id", "year", "i_sample"], ignore_index=True))
        path = out_dir / f"{feature}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths
