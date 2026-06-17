import os
import gc
import importlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


import epiweeks
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from inframind_proteus.empirical_data import DiseaseTimeSeriesCache
from inframind_proteus.outbreak_dynamics import RenewalSimulator, build_calibration_params_df, build_initial_infec_df, \
    SimulationConfig
from inframind_proteus.outbreak_dynamics.sampling import GammaPrior
from inframind_proteus.outbreak_dynamics.utils import (
    load_yaml_dict, apply_include_exclude_logic, map_parallel_or_sequential_chunks, save_yaml_dict,
    map_parallel_or_sequential
)

class ProgramConfig:

    base_config_fpath: Path = Path("configs/prototype_run_dynamic_model.yaml")
    uf_table_fpath = Path("data/demographic/uf_table.csv")

    # --- Temporal config, epiweek-based (year-agnostic)
    zero_date_epiweek = 41
    sim_start_epiweek = 26  # Note: Train data for each season ends at epiweek 25.
    calibration_start_epiweek = 41
    calibration_end_epiweek = 40  # Of the next year

    # --- Locations and years to run
    # use_location_ids = ["SP", "SE", "MG", "PA"]  # Runs all!
    use_location_ids = ["SP"]  # Runs all!
    exclude_location_ids = [] # ["CE"]  # Just to test.
    use_years = list(range(2022, 2023))
    ncpus = 1

    # ---
    stage1_num_simulations = 2**20

    stage2_num_simulations = 2**18
    stage2_free_params = [
        "rt_logist_r_high",
        "rt_logist_start",
        # "notif_relative_scale",
    ]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main():

    cfg = ProgramConfig()
    pass

    # --- Load global auxiliary files
    uf_table_df = pd.read_csv(cfg.uf_table_fpath)
    base_config_dict = load_yaml_dict(cfg.base_config_fpath)

    # --- Combine all years and locations to be run
    location_ids = apply_include_exclude_logic(
        uf_table_df["uf"],
        include_list=cfg.use_location_ids,
        exclude_list=cfg.exclude_location_ids,
    )
    years = list(cfg.use_years)
    location_year_index = pd.MultiIndex.from_product(
        [location_ids, years],
        names=["uf", "season"]
    )

    # ---- (test) Sequential loop over location-year combinations
    def _task(location_year_tuple):
        run_calibration_for_location_year(
            location_year_tuple,
            cfg=cfg,
            base_config_dict=base_config_dict,
            uf_table_df=uf_table_df,
        )

    results = map_parallel_or_sequential(
        _task, location_year_index, ncpus=cfg.ncpus,
    )


    # for location_year_tuple in location_year_index:
    #     run_calibration_for_location_year(
    #         location_year_tuple,
    #         cfg=cfg,
    #         base_config_dict=base_config_dict,
    #         uf_table_df=uf_table_df,
    #     )

    assert True


def run_calibration_for_location_year(
        location_year_tuple,
        cfg: ProgramConfig,
        base_config_dict: dict,
        uf_table_df: pd.DataFrame,
):
    location_id, year = location_year_tuple
    print(f"run_calibration_for_location_year(({location_id} in {year}))")

    # # Load again observations, since this function will run in parallel... not worth caching
    # obserations_sr = DiseaseTimeSeriesCache(Path("data/disease/dengue_cases_uf_weekly")).get(location_id)
    # Load observations
    observations_sr = DiseaseTimeSeriesCache().get_location(location_id)

    # Stage 1: Broad parameter exploration with max. likelihood selection
    # ===============
    run_calibration_stage_1(
        location_id, year,
        cfg=cfg,
        base_config_dict=base_config_dict,
        observations_sr=observations_sr,
        uf_table_df=uf_table_df,
    )
    assert True




def _yw_to_date_str(year, week):
    return epiweeks.Week(year, week).startdate().isoformat()


def _calc_scaling_from_presim_period(
        sim_cfg: SimulationConfig,
        observations_sr: pd.Series,
):
    # --- Fetch incidence at pre-simulation phase
    # presim_start_date = config.temporal.sim_start - pd.Timedelta(simulator._gt_max_steps, unit="W")
    presim_start_date = sim_cfg.temporal.sim_start - pd.Timedelta(6, unit="W")
    presim_end_date = sim_cfg.temporal.sim_start - pd.Timedelta(1, unit="W")


    # Calculate the relative scaling factor to
    # OBS: Could directly calculate the scaling factor, but it's less interpretable and harder to bound.
    # Assumes initialization with method "ones" (infections = 1)
    coef = (
            sim_cfg.observation_model.reference_population_size
            / sim_cfg.location.population_size
            / 1.
    )

    # Observation series at the pre-sampling window
    presamp_obs_sr = observations_sr.loc[presim_start_date:presim_end_date]
    mean_rel_scaling = presamp_obs_sr.mean() * coef
    std_rel_scaling = presamp_obs_sr.std() * coef

    return mean_rel_scaling, std_rel_scaling


@dataclass
class Stage1Outputs:
    max_ll_params: pd.Series


def run_calibration_stage_1(
        location_id,
        year,
        cfg: ProgramConfig,
        base_config_dict: dict,
        observations_sr: pd.Series,
        uf_table_df: pd.DataFrame,
):
    print(f"run_calibration_stage_1(({location_id}, {year}))")

    observations_sr = DiseaseTimeSeriesCache().get_location(location_id)
    config_dict = _d = deepcopy(base_config_dict)

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
    _temporal["zero_date"] = _yw_to_date_str(year, cfg.zero_date_epiweek)
    _temporal["sim_start"] = _yw_to_date_str(year, cfg.sim_start_epiweek)
    _temporal["calibration_start"] = _yw_to_date_str(year, cfg.calibration_start_epiweek)
    _temporal["calibration_end"] = _yw_to_date_str(year + 1, cfg.calibration_end_epiweek)

    _d["simulation"]["num_simulations"] = cfg.stage1_num_simulations

    _scoring = _d["scoring"]
    _scoring["metrics"] = ["nb_loglikelihood"]

    # --- Create simulator object with modified configuration dictionary
    simulator = RenewalSimulator.from_config_dict(config_dict)
    sim_cfg = simulator.config

    # --- Calculate scaling factor priors from pre-simulation period observations
    mean_rel_scaling, std_rel_scaling = _calc_scaling_from_presim_period(
        sim_cfg, observations_sr
    )
    sim_cfg.sampling.param_priors["notif_relative_scale"] = GammaPrior(
        mean_rel_scaling, std_rel_scaling
    )

    # --- Build auxiliary data frames for simulations
    params_df = build_calibration_params_df(sim_cfg.num_simulations, sim_cfg.sampling)
    sim_cfg.num_simulations = num_simulations = params_df.shape[0]  # Update in case sampling method changes number of simulations
    initial_infec_df = build_initial_infec_df(
        sim_cfg.num_simulations,
        simulator._gt_max_steps,
        sim_cfg.temporal.step_dt,
        sim_cfg.initial_infections
    )


    # ====
    # --- RUN
    results = simulator.run_sequential_chunks(
        params_df=params_df,
        initial_infec_df=initial_infec_df,
        observations_sr=observations_sr,
    )

    del initial_infec_df
    gc.collect()

    # ====
    # --- Get maximum likelihood simulation

    i_max = results.scoring.summary["nb_loglikelihood"].idxmax()
    max_ll_params = params_df.loc[i_max]

    # ===
    # --- Visualization/diagnostics
    dir = Path(f".local/calibrate_multistage/{location_id}_{year}")
    dir.mkdir(exist_ok=True, parents=True)

    # --- Plot maximum likelihood case beam along with data
    rc = deepcopy(plt.rcParams)
    rc["patch.linewidth"] = 0
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8, 4))

        # Observations
        _obs = observations_sr.loc[sim_cfg.temporal.sim_start:sim_cfg.temporal.calibration_end]
        ax.plot(_obs.index, _obs.values, "ks")

        # Simulation median and quantiles
        _df = results.case_beam_df.xs(i_max, level="i_simulation")
        ax.fill_between(
            _df.columns,
            _df.loc[0.025], _df.loc[0.975],
            color="darkslateblue", alpha=0.3
        )
        ax.fill_between(
            _df.columns,
            _df.loc[0.25], _df.loc[0.75],
            color="darkslateblue", alpha=0.3
        )
        ax.plot(_df.loc[0.5], color="darkslateblue")

        fig.tight_layout()

        fig.savefig(dir / "stage1_max_ll_case_beam.pdf")

    # ====

    out = Stage1Outputs(
        max_ll_params=max_ll_params,
    )

    return out





if __name__ == "__main__":
    main()
