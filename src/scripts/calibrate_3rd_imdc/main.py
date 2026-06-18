"""
Calibration procedure of the outbreak dynamics model (renewal equation) of
Inframind Proteus.

This script is originally developed for the validation round of the
3rd Infodengue-Mosqlimate Dengue Challenge (3rd IMDC).

Usage
-----

"""

from __future__ import annotations

import io
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from inframind_proteus import BaseConfig
from inframind_proteus.outbreak_dynamics import SimulationConfig
from inframind_proteus.outbreak_dynamics.utils import (
    parse_set_arguments_with_yaml, load_yaml_dict, year_week_to_date,
    apply_include_exclude_logic, map_parallel_or_sequential
)
from scripts.calibrate_3rd_imdc.program_config import ProgramConfig


def parse_args_get_dict(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse command-line arguments and return them as a dictionary."""
    parser = ArgumentParser()


    # --- Config file path
    parser.add_argument(
        "--config-fpath", "--cfg", "-c",
        default=ProgramConfig.config_fpath,
        type=Path,
        help="Path to the calibration YAML configuration file.",
    )

    # --- Generic nested `--set` argument.
    parser.add_argument(
        "--set",
        nargs=2,
        action="append",
        default=list(),
        metavar=("KEY", "VALUE"),
        help="Set a configuration parameter using dot notation (e.g. "
             "--set stage1.num_simulations 4096)."
             " Can be used multiple times.",
    )


    # ======

    args = parser.parse_args(argv)
    # Retain only informed arguments to avoid overriding config.
    args_dict = {k: v for k, v in args.__dict__.items() if v is not None}

    # Proceess --set arguments to override config values.
    set_args = args_dict.pop("set")
    overrides = parse_set_arguments_with_yaml(set_args)
    args_dict.update(overrides)

    return args_dict


class ProgramData:
    """Internal payload data class for the calibration procedure script."""
    uf_table_df: pd.DataFrame
    base_sim_config_dict: dict[str, Any]
    base_sim_config: SimulationConfig
    # Note: The sim config is a payload because it is not fully compatible with
    #   the BaseConfig scheme, as it runs through a custom validation `from_dict()`.
    #   If this gets fixed, one could override params through command line.

    location_ids: list
    years: list
    location_year_index: pd.MultiIndex


def main(argv: list[str] | None = None) -> None:

    # --- Program initialization sequence
    args_dict = parse_args_get_dict(argv)
    config_dict = load_yaml_dict(args_dict["config_fpath"])
    cfg = ProgramConfig()
    cfg.update_from_dict(config_dict)
    cfg.update_from_dict(args_dict)
    cfg.preprocess()

    # --- Load basic payload
    data = ProgramData()
    data.uf_table_df = pd.read_csv(cfg.uf_table_fpath)
    data.base_sim_config_dict = load_yaml_dict(cfg.base_sim_config_fpath)

    # --- Combine all years and locations to be run
    data.location_ids = apply_include_exclude_logic(
        data.uf_table_df["uf"],
        include_list=cfg.use_location_ids,
        exclude_list=cfg.exclude_location_ids,
    )
    data.years = list(cfg.use_years)
    data.location_year_index = pd.MultiIndex.from_product(
        [data.location_ids, data.years],
        names=["uf", "season"]
    )

    # --- Run calibration algorithm for all location-year combinations
    def _task(location_year_tuple):
        location_id, year = location_year_tuple
        run_calibration_stages(
            location_id, year,
            cfg=cfg,
            base_sim_config_dict=data.base_sim_config_dict,
            uf_table_df=data.uf_table_df,
        )
    _contents = data.location_year_index
    map_parallel_or_sequential(
        _task, _contents,
        ncpus=cfg.ncpus,
        chunksize=1
    )


# Internal auxiliary functions
# =========================================
def _set_config_dict_common(
        cfg: ProgramConfig,
        sim_config_dict: dict,
        location_id,
        year,
        uf_table_df: pd.DataFrame
):
    """Override simulation configuration dictionary fields with parameters
    that are common to all calibration stages.
    """
    _d = sim_config_dict

    def _todate(y, w):
        return year_week_to_date(y, w).date().isoformat()

    # --- Set location-specific config fields
    _d["location"]["location_id"] = location_id
    _d["location"]["population_size"] = (
        uf_table_df
        .set_index("uf")
        .loc[location_id, f"population_{year}"]
        .item()
    )

    # --- Set the time fields (simulation start date, calibration window, etc)
    _temporal = _d["temporal"]
    _temporal["zero_date"] = _todate(year, cfg.zero_date_epiweek)
    _temporal["sim_start"] = _todate(year, cfg.sim_start_epiweek)
    _temporal["calibration_start"] = _todate(year, cfg.calibration_start_epiweek)
    _temporal["calibration_end"] = _todate(year + 1, cfg.calibration_end_epiweek)


# Main routines
# =========================

def run_calibration_stages(
        location_id, year,
        cfg: ProgramConfig,
        base_sim_config_dict: dict,
        uf_table_df: pd.DataFrame,
):
    """"""

    pass

if __name__ == "__main__":
    main(sys.argv[1:])
