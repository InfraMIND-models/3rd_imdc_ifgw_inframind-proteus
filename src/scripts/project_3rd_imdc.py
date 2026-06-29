"""Run projections with the outbreak dynamic model for unknown data."""

from __future__ import annotations

import io
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Tuple

import matplotlib as mpl
import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from inframind_proteus import BaseConfig
from inframind_proteus.empirical_data import DiseaseTimeSeriesCache
from inframind_proteus.outbreak_dynamics import SimulationConfig, RenewalSimulator
from inframind_proteus.outbreak_dynamics.simulator import sample_negative_binomial_trajectories
from inframind_proteus.outbreak_dynamics.utils import (
    parse_set_arguments_with_yaml, load_yaml_dict, year_week_to_date,
    apply_include_exclude_logic, map_parallel_or_sequential, make_yaml_exportable_dict, save_yaml_dict, add_set_argument
)



def main(argv: Union[list[str], None] = None):
    cfg, data = initialize_program(argv)

    load_and_preprocess_global_data(cfg, data)

    def _task(location_year_tuple):
        location_id, year = location_year_tuple
        run_projections_for_location_year(cfg, data, location_id, year)

    _contents = data.location_year_index

    map_parallel_or_sequential(
        _task, _contents, ncpus=cfg.ncpus
    )

    return


class ProgramConfig(BaseConfig):
    """Internal configuration data class for the
    `process_data_for_projections` script.
    """
    config_fpath: Path = Path("configs/project_3rd_imdc_default.yaml")
    base_sim_config_fpath: Path = Path("configs/simulation_config_default.yaml")
    # calibrations_main_dir: Path = Path("outputs/validation_round_calibration")
    uf_table_fpath = Path("data/demographic/uf_table.csv")

    # --- Output directory subpaths
    output_dir: Path = Path("outputs/validation_round_projections")
    projections_dirname: str = "projections"
    location_year_subdir_fmt: str = "{location_id}_{year}"
    parameter_samples_fname: str = (
        "projection_parameter_samples.csv"
    )  #  ^ Expected to be within the location-year subdirectory of output_dir

    # --- Temporal config, epiweek-based (year-agnostic)
    zero_date_epiweek: int = 41  # Reference date for t = 0 (not simulation start)
    sim_start_epiweek: int = 26  # Note: Train data for each season ends at epiweek 25.

    # --- Locations and years to run
    use_location_ids = []  # Runs all!
    exclude_location_ids: list[str] = list()
    use_years = [2022, 2023, 2024, 2025]

    ncpus = 1

    # Split simulations in sequential chunks.
    #   Use lower values to reduce RAM usage.
    #   Use None to keep default
    simulator_max_chunk_size: int | None = None


class ProgramData:
    """Internal payload data class for the projections procedure script."""
    uf_table_df: pd.DataFrame
    base_sim_config_dict: dict[str, Any]
    # base_sim_config: SimulationConfig
    # Note: The sim config is a payload because it is not fully compatible with
    #   the BaseConfig scheme, as it runs through a custom validation `from_dict()`.
    #   If this gets fixed, one could override params through command line.

    location_ids: list
    years: list
    location_year_index: pd.MultiIndex

    # --- Location and year-specific data
    parameter_samples_df: pd.DataFrame
    # Similar to `posterior_samples_df` in `process_data_for_projections`


def parse_args_get_dict(argv) -> dict:
    """"""
    parser = ArgumentParser()

    # --- Config file path
    parser.add_argument(
        "--config-fpath", "--cfg", "-c",
        default=ProgramConfig.config_fpath,
        type=Path,
        help="Path to the calibration YAML configuration file.",
    )

    parser.add_argument(
        "--output-dir", "--out", "-o",
        default=ProgramConfig.output_dir,
        type=Path,
        help="Path to the output directory.",
    )

    parser.add_argument(
        "--use-location-ids",  "-l",
        default=None,
        type=str,
        action="append",
        help=(
            "List of location IDs to process. Can be specified multiple times. "
            "Using this argument resets the default list from the program or "
            "config file. Example: \"-l SP -l MG\" will set use_location_ids to "
            "['SP', 'MG'], regardless of what other locations have been specified"
            "on the defaults."
        ),
    )

    parser.add_argument(
        "--use-years",  "-y",
        default=None,
        type=int,
        action="append",
        help=(
            "List of years to prepare projections for. "
            "Can be specified multiple times. "
            "Using this argument resets the default list from the program or "
            "config file. Example: \"-y 2022 -y 2023\" will set use_years to "
            "[2022, 2023], regardless of what other years have been specified"
            "on the defaults."
        )
    )

    # --- Generic nested `--set` argument.
    add_set_argument(parser)

    # ======

    args = parser.parse_args(argv)
    # Retain only informed arguments to avoid overriding config.
    args_dict = {k: v for k, v in args.__dict__.items() if v is not None}

    # Proceess --set arguments to override config values.
    set_args = args_dict.pop("set")
    overrides = parse_set_arguments_with_yaml(set_args)
    args_dict.update(overrides)

    return args_dict


def initialize_program(argv) -> Tuple[ProgramConfig, ProgramData]:
    # --- Program initialization sequence
    args_dict = parse_args_get_dict(argv)
    cfg = ProgramConfig()
    if "config_fpath" in args_dict:
        config_dict = load_yaml_dict(args_dict["config_fpath"])
        cfg.update_from_dict(config_dict)
    cfg.update_from_dict(args_dict)
    cfg.preprocess()
    data = ProgramData()

    return cfg, data


def load_and_preprocess_global_data(cfg: ProgramConfig, data: ProgramData):
    """Load global data that is not location/year specific."""
    # --- Load UF table
    data.uf_table_df = pd.read_csv(cfg.uf_table_fpath)

    # --- Load base simulation config to retrieve parameters from
    data.base_sim_config_dict = load_yaml_dict(cfg.base_sim_config_fpath)
    # data.base_sim_config = SimulationConfig.from_dict(data.base_sim_config_dict)

    # --- Set lists of contents to iterate on
    data.location_ids = apply_include_exclude_logic(
        all_series=data.uf_table_df["uf"],
        include_list=cfg.use_location_ids,
        exclude_list=None,
    )
    data.years = cfg.use_years
    data.location_year_index = pd.MultiIndex.from_product(
        [data.location_ids, data.years],
        names=["uf", "season"]
    )


def run_projections_for_location_year(
        cfg: ProgramConfig,
        data: ProgramData,
        location_id: str,
        year: int
):
    """Run projections for a specific location and year."""
    print(f"Running projections for {location_id} in {year}...")
    uf_table_df = data.uf_table_df
    out_dir = (
            cfg.output_dir
            / cfg.projections_dirname
            / cfg.location_year_subdir_fmt.format(
                location_id=location_id, year=year
            )
    )
    # out_dir.mkdir(parents=True, exist_ok=True)
    # In this case, the out dir must already exist since it should contain
    # the projection parameter samples file.
    if not out_dir.exists():
        raise FileNotFoundError(
            f"Output directory not found: {out_dir}. "
            f"It should exist and contain supporting data to run projections."
        )

    # Override simulation config with location and year specific parameters
    # ==============
    _d = sim_config_dict = deepcopy(data.base_sim_config_dict)

    def _todate(y, w):
        return year_week_to_date(y, w).date().isoformat()

    _d["simulation"]["mode"] = "projection"

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

    # --- Modify sampling (no sampling, parameters are given)
    _sampling = _d["sampling"]
    _sampling["method"] = "given"

    # Load pre-calculated parameter samples
    # =======================
    _fpath = out_dir / cfg.parameter_samples_fname
    if not _fpath.exists():
        raise FileNotFoundError(
            f"Projection parameter samples file not found: {_fpath}. Perhaps "
            f"you need to run the calibration and processing scripts first."
        )

    data.parameter_samples_df = pd.read_csv(
        _fpath, index_col="i_simulation"
    )

    # quick validations
    df = data.parameter_samples_df
    if not (df.index == pd.RangeIndex(df.shape[0])).all():
        raise ValueError()

    # Simulator and simulations
    # ==============
    # --- Create simulator object with modified configuration dictionary
    simulator = RenewalSimulator.from_config_dict(sim_config_dict)
    sim_cfg = simulator.config

    # --- Run the simulation in projections mode
    _kwargs = dict()
    if cfg.simulator_max_chunk_size is not None:
        _kwargs["max_chunk_size"] = cfg.simulator_max_chunk_size

    params_df, initial_infec_df = (
        simulator.build_simulation_data(
            sampling_kwargs=dict(
                given_params=data.parameter_samples_df
            )
        )
    )

    sim_results = simulator.run_sequential_chunks(
        params_df=params_df,
        initial_infec_df=initial_infec_df,
        **_kwargs
    )

    # Sample trajectories of observed cases
    # ===========

    # For now we will assume the negative binomial model only.
    if sim_cfg.observation_model.model != "negative_binomial":
        raise ValueError(
            "The observation model is not negative binomial. "
            "This procedure is currently hardcoded for that model only."
        )

    # Use the simulation seed to take samples
    rng = np.random.default_rng(sim_cfg.rng_seed)

    cases_vec = (
        sample_negative_binomial_trajectories(
            expectancy=sim_results.mean_cases_df.to_numpy(),
            overdisp=params_df["notif_nb_overdispersion"].to_numpy(),
            rng=rng
        )
    )
    cases_df = pd.DataFrame(
        cases_vec,
        index=sim_results.mean_cases_df.index,
        columns=sim_results.mean_cases_df.columns,
    )

    # Just check it
    fig, ax = plt.subplots()
    ax.plot(
        cases_df.T,
        alpha=0.1,
        color="blue",
    )
    fig.show()

    return


if __name__ == "__main__":
    main(sys.argv[1:])

