import gc
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
from matplotlib import pyplot as plt

from inframind_proteus.outbreak_dynamics import RenewalSimulator, SimulationConfig, SimulationOutput
from inframind_proteus.outbreak_dynamics.sampling import GammaPrior
from .helpers import _set_config_dict_common, prepare_output_subdirs
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
) -> Stage1Outputs:
    """"""
    print(f"\trun_calibration_stage_1({location_id}, {year})")
    stage1_cfg = cfg.stage1

    # Preamble
    # =====================

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

    # --- Adjust outputs and retained data
    # False to save some memory
    sim_config_dict["output"]["keep_rt_trajectories"] = False

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


    # Simulations
    # ==================
    # --- Run the simulation and scoring
    _kwargs = dict()
    if cfg.simulator_max_chunk_size is not None:
        _kwargs["max_chunk_size"] = cfg.simulator_max_chunk_size

    params_df, initial_infec_df = (
        simulator.build_simulation_data()
    )
    sim_results = simulator.run_sequential_chunks(
        params_df=params_df,
        initial_infec_df=initial_infec_df,
        observations_sr=observations_sr,
        **_kwargs
    )
    gc.collect()


    # Simulation postprocessing
    # =========================
    # --- Get maximum likelihood simulation and its parameters
    i_max = sim_results.scoring.summary["nb_loglikelihood"].idxmax()
    max_ll_params = params_df.loc[i_max]
    max_ll_params.index.name = "parameter_name"
    max_ll_params.name = "value"

    # -----------------

    _stage_1_plots_and_diagnostics(
        location_id, year,
        cfg=cfg,
        simulator=simulator,
        sim_results=sim_results,
        observations_sr=observations_sr,
        uf_table_df=uf_table_df,
        max_ll_params=max_ll_params,
        i_max=i_max,
    )

    return Stage1Outputs(
        max_ll_params=max_ll_params
    )




# Internal helpers
# =================

def _calc_scaling_from_presim_period(
        sim_cfg: SimulationConfig,
        observations_sr: pd.Series,
        presim_period_num_points,
        force_nonzero=True

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

    presim_mean = presim_obs_sr.mean()
    presim_std = presim_obs_sr.std()

    # Calculate the relative scaling factor to match average recent observations
    # OBS: Could directly calculate the scaling factor, but it's less interpretable and harder to bound.
    # Assumes initialization with method "ones" (infections = 1)
    coef = (
            sim_cfg.observation_model.reference_population_size
            / sim_cfg.location.population_size
            / 1.
    )

    # Regularize small case counts to avoid zero mean and std
    if force_nonzero:
        # Minimum: At least one case in the pre-simulation period
        thresh = 1. / presim_obs_sr.shape[0]
        if presim_mean < thresh:
            presim_mean = thresh
        if presim_std < thresh:
            # Set also STD to avoid a collapsed gamma prior
            presim_std = thresh

    # Observation series at the pre-sampling window
    mean_rel_scaling = presim_mean * coef
    std_rel_scaling = presim_std * coef

    return mean_rel_scaling, std_rel_scaling


def _stage_1_plots_and_diagnostics(
        location_id, year,
        cfg: ProgramConfig,
        simulator: RenewalSimulator,
        sim_results: SimulationOutput,
        # base_config_dict: dict,
        observations_sr: pd.Series,
        uf_table_df: pd.DataFrame,
        max_ll_params: pd.Series = None,
        i_max: int = None
):
    """"""
    stage1_cfg = cfg.stage1
    sim_cfg = simulator.config
    data_out_dir, plots_out_dir = prepare_output_subdirs(
        location_id, year,
        output_dir=cfg.output_dir,
        location_year_subdir_fmt=cfg.location_year_subdir_fmt,
        mkdirs=True
    )

    rc = deepcopy(plt.rcParams)
    rc["patch.linewidth"] = 0
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8, 4))

        # Observations (within calibration period)
        _obs = observations_sr.loc[
            sim_cfg.temporal.sim_start
            :sim_cfg.temporal.calibration_end
        ]
        ax.plot(_obs.index, _obs.values, "ks")

        # Simulation median and quantiles
        _df: pd.DataFrame = (
            sim_results.case_beam_df.xs(
                i_max, level="i_simulation"
            )
        )
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

        ax.set_ylabel("Weekly cases")
        fig.tight_layout()

        fig.savefig(plots_out_dir / "stage1_max_ll_case_beam.pdf")
        plt.close(fig)

    # --- Export some stuff
    # TODO: Move to export helper
    max_ll_params.to_csv(data_out_dir / "stage1_max_ll_params.csv")
