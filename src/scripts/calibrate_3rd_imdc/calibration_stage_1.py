from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from inframind_proteus.outbreak_dynamics import RenewalSimulator, SimulationConfig
from inframind_proteus.outbreak_dynamics.sampling import GammaPrior
from .helpers import _set_config_dict_common
from .program_config import ProgramConfig


@dataclass
class Stage1Outputs:
    max_ll_params: pd.Series


def run_calibration_stage_1(
        location_id, year,
        cfg: ProgramConfig,
        base_sim_config_dict: dict,
        observations_sr: pd.Series,
        uf_table_df: pd.DataFrame,
):
    """"""
    print(f"\trun_calibration_stage_1({location_id}, {year})")
    stage1_cfg = cfg.stage1

    # --- Instantiate simulation dictionary for this round
    sim_config_dict = deepcopy(base_sim_config_dict)
    _set_config_dict_common(
        cfg, sim_config_dict,
        location_id,
        year,
        uf_table_df,
        num_simulations=stage1_cfg.num_simulations,
        scoring_metrics=[
            "nb_loglikelihood",
        ]
    )

    # --- Manually remove overdispersion from exploration (adjusted later in stage 3)
    _sampling = sim_config_dict["sampling"]
    if "notif_nb_overdispersion" in _sampling["param_ranges"]:
        del _sampling["param_ranges"]["notif_nb_overdispersion"]

    # --- Create simulator object with modified configuration dictionary
    simulator = RenewalSimulator.from_config_dict(sim_config_dict)
    sim_cfg = simulator.config

    # --- Calculate scaling factor priors from pre-simulation period observations
    mean_rel_scaling, std_rel_scaling = _calc_scaling_from_presim_period(
        sim_cfg, observations_sr, stage1_cfg.presim_period_num_points
    )
    sim_cfg.sampling.param_priors["notif_relative_scale"] = GammaPrior(
        mean_rel_scaling, std_rel_scaling
    )

    # --- Run the simulation and scoring
    _kwargs = dict()
    if cfg.simulator_max_chunk_size is not None:
        _kwargs["max_chunk_size"] = cfg.simulator_max_chunk_size

    params_df, initial_infec_df = (
        simulator.build_simulation_data()
    )
    simulator.run_sequential_chunks(
        params_df=params_df,
        initial_infec_df=initial_infec_df,
        observations_sr=observations_sr,
        **_kwargs
    )



# Internal helpers
# =================

def _calc_scaling_from_presim_period(
        sim_cfg: SimulationConfig,
        observations_sr: pd.Series,
        presim_period_num_points,
):
    # --- Fetch incidence at pre-simulation phase
    _n = presim_period_num_points
    # presim_start_date = config.temporal.sim_start - pd.Timedelta(simulator._gt_max_steps, unit="W")
    presim_start_date = (
            sim_cfg.temporal.sim_start
            - pd.Timedelta(_n, unit="W")
    )
    presim_end_date = (
            sim_cfg.temporal.sim_start
            - pd.Timedelta(1, unit="W")
    )
    presim_obs_sr = (
        observations_sr
        .loc[presim_start_date:presim_end_date]
    )

    # Calculate the relative scaling factor to match average recent observations
    # OBS: Could directly calculate the scaling factor, but it's less interpretable and harder to bound.
    # Assumes initialization with method "ones" (infections = 1)
    coef = (
            sim_cfg.observation_model.reference_population_size
            / sim_cfg.location.population_size
            / 1.
    )

    # Observation series at the pre-sampling window
    mean_rel_scaling = presim_obs_sr.mean() * coef
    std_rel_scaling = presim_obs_sr.std() * coef

    return mean_rel_scaling, std_rel_scaling


