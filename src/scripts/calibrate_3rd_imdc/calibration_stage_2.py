import gc
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from inframind_proteus.outbreak_dynamics import RenewalSimulator, SimulationConfig, SimulationOutput
from inframind_proteus.outbreak_dynamics.sampling import GammaPrior
from .helpers import _set_config_dict_common, prepare_output_subdirs
from .program_config import ProgramConfig
from .calibration_stage_1 import Stage1Outputs


@dataclass
class Stage2Outputs:
    param_samples: pd.DataFrame
    posterior_kde: gaussian_kde
    kde_param_names: list[str]


def run_calibration_stage_2(
        location_id, year,
        cfg: ProgramConfig,
        base_sim_config_dict: dict,
        observations_sr: pd.Series,
        uf_table_df: pd.DataFrame,
        stage1_outputs: Stage1Outputs,
):
    """"""
    print(f"\trun_calibration_stage_2({location_id}, {year})")

    # Preamble
    # =====================

    # --- Instantiate simulation dictionary for this round
    sim_config_dict = deepcopy(base_sim_config_dict)
    _set_config_dict_common(
        cfg, sim_config_dict,
        location_id,
        year,
        uf_table_df,
        num_simulations=cfg.stage2.num_simulations,
        scoring_metrics=[
            "nb_loglikelihood",
        ]
    )

    # --- Override parameters found in stage 1 and remove from exploration
    param_ranges = sim_config_dict["sampling"]["param_ranges"]
    _d = sim_config_dict
    _sampling = _d["sampling"]
    nuisance_param_names = [
        p for p in param_ranges.keys()
        if p not in cfg.stage2.free_params
    ]
    for param_name in nuisance_param_names:
        # Set max likelihood value from stage 1 as fixed value for this stage
        # (Try to guess parameter location from its name)
        if param_name.startswith("rt_"):
            sub_d = _d["reproduction_number"]["params"]
        elif param_name.startswith("notif_"):
            sub_d = _d["observation_model"]["params"]
        else:
            raise ValueError(f"Cannot guess parameter location for {param_name}")
        sub_d[param_name] = stage1_outputs.max_ll_params[param_name]

        # Remove from exploration
        del _sampling["param_ranges"][param_name]

    # --- Adjust outputs and retained data
    # True so we can plot Rt trajectories
    # False will save memory
    sim_config_dict["output"]["keep_rt_trajectories"] = True

    # --- Create simulator object with modified configuration dictionary
    simulator = RenewalSimulator.from_config_dict(sim_config_dict)
    sim_cfg = simulator.config

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
    post_samples_df, post_kde = (
        _postprocess_stage_2_simulations(
            cfg, params_df, sim_results
        )
    )

    _stage_2_plots_and_diagnostics(
        location_id, year,
        cfg,
        observations_sr,
        params_df,
        post_kde,
        post_samples_df,
        sim_cfg,
        sim_results
    )

    return Stage2Outputs(
        param_samples=post_samples_df,
        posterior_kde=post_kde,
        kde_param_names=cfg.stage2.free_params,
    )


# Internal helpers
# =================


def _postprocess_stage_2_simulations(
        cfg: ProgramConfig,
        params_df: pd.DataFrame,
        sim_results: SimulationOutput
) -> tuple[pd.DataFrame, gaussian_kde]:

    # --- Build posterior distributions
    rng = np.random.default_rng(cfg.stage2.posterior_seed)
    # Treat the likelihood function
    df = pd.concat([params_df, sim_results.scoring.summary], axis=1)
    tempered_ll = df["nb_loglikelihood"] / cfg.stage2.ll_temperature
    tempered_ll -= tempered_ll.max()  # Displace by max
    df["tempered_likelihood"] = np.exp(tempered_ll)

    # --- Apply prior and combine
    # df["prior_weight"] = cfg.stage2.prior(df)  # Disabled for now
    df["prior_weight"] = 1.  # cfg.stage2.prior(df)
    df["posterior_weight"] = df["tempered_likelihood"] * df["prior_weight"]

    # --- Regularize small weights
    max_weight = df["posterior_weight"].max()
    w_cutoff = max_weight * cfg.stage2.rel_weight_cutoff
    num_relevant_weights = (df["posterior_weight"] >= w_cutoff).sum()

    # --- Decision tree: Number of samples to keep based on surviving weights
    min_samples = cfg.stage2.min_samples_to_kde
    max_samples = cfg.stage2.max_samples_to_kde
    if num_relevant_weights < min_samples:
        # Keep min number or as many as available on the original sample size
        df = (
            df
            .sort_values("posterior_weight")
            .iloc[-min_samples:]
        )
    elif num_relevant_weights < max_samples:
        # Keep all above cutoff
        df = (
            df[df["posterior_weight"] >= w_cutoff]
            .sort_values("posterior_weight")
        )
    else:  # num_relevant_weights >= max_samples
        # Re-sample from weights above threshold to keep max number
        df = (
            df[df["posterior_weight"] >= w_cutoff]
            .sample(
                n=max_samples,
                replace=False,
                weights=None,  # No weights when just reducing nr. of samples
                random_state=rng,
            )
            .sort_values("posterior_weight")
        )
    # At this point, df contains selected samples and their weights, with a
    #    variable size between specified min and max.
    post_samples_df = df

    # from scipy.stats import gaussian_kde
    post_kde = gaussian_kde(
        post_samples_df[cfg.stage2.free_params].T.values,
        weights=post_samples_df["posterior_weight"]
    )

    return post_samples_df, post_kde


def _stage_2_plots_and_diagnostics(
        location_id, year,
        cfg, observations_sr, params_df, post_kde, post_samples_df,
        sim_cfg, sim_results
):
    data_out_dir, plots_out_dir = prepare_output_subdirs(
        location_id, year,
        output_dir=cfg.output_dir,
        location_year_subdir_fmt=cfg.location_year_subdir_fmt,
        mkdirs=True
    )

    # --- Visualize median trajectories against data
    df = post_samples_df
    kde = post_kde
    rc = deepcopy(plt.rcParams)
    # Sample pre-simulated trajectories with posteriors as weights
    sampled_params_df: pd.DataFrame = params_df.sample(
        n=500, weights=df["posterior_weight"], replace=True, random_state=123,
    )
    sampled_beams = sim_results.case_beam_df.loc[
        sim_results.case_beam_df.index.get_level_values("i_simulation").isin(sampled_params_df.index)
    ]
    # Reproduction number from the same sampled trajectories
    # Rebuild the full params df but only with the best sampled parameters
    # rt_df = _rebuild_rt_curves(results.config, sampled_params_df, simulator)
    rt_df = sim_results.rt_df

    if rt_df is None:
        print("Warning: Rt trajectories not available for stage 2 plots.")
        rt_df = sampled_rt_df = pd.DataFrame()  # Empty dataframe to avoid errors in plotting
    else:
        sampled_rt_df = rt_df.reindex(sampled_params_df.index)

    rc = deepcopy(plt.rcParams)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrows=2, figsize=(8, 7))

        # --- Case Trajectories
        ax = axes[0]

        # Observations
        _obs = observations_sr.loc[sim_cfg.temporal.sim_start:sim_cfg.temporal.calibration_end]
        ax.plot(_obs.index, _obs.values, "ks", label="Observations")

        # Sampled trajectories
        ax.plot(
            sampled_beams.xs(0.5, level="quantile").T,
            color="darkslateblue", alpha=0.2,
        )

        ax.set_ylabel("Weekly cases")
        ax.legend()

        # --- Reproduction number
        _df = sampled_rt_df
        ax = axes[1]
        ax.plot(_df.T, color="darkslateblue", alpha=0.2)
        ax.plot(_df.columns, np.ones(_df.shape[1]), "k--", label="R=1")
        ax.set_ylabel("Reproduction number")
        ax.set_xlim(*axes[0].get_xlim())  # Align x-axis with case trajectories

        fig.tight_layout()
        fig.savefig(plots_out_dir / "stage2_median_trajectories.pdf")
        plt.close(fig)

    # --- Visualize posteriors
    num_params = len(cfg.stage2.free_params)
    param_ranges = sim_results.config.sampling.param_ranges
    rc = deepcopy(plt.rcParams)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows=num_params, ncols=1,
            figsize=(8, num_params * 3)
        )

        for i, param_name in enumerate(cfg.stage2.free_params):
            ax: plt.Axes = axes[i]

            # Plot posterior distribution
            # _df = df[df["posterior_weight"] > 0]
            ax.hist(
                df[param_name],
                bins=150,
                range=param_ranges[param_name],
                weights=df["posterior_weight"],
                label="Posterior samples",
                color="darkslateblue",
                density=True,
            )

            # Plot posterior KDE
            x = np.linspace(df[param_name].min(), df[param_name].max(), 100)
            m_kde = kde.marginal(i)
            y = m_kde.evaluate(x)
            ax.plot(x, y, color="tomato", lw=1, label="Marginal KDE")

            ax.set_xlim(*param_ranges[param_name])
            ax.set_xlabel(param_name)
            ax.set_ylabel("Density")

        axes[0].legend()
        fig.tight_layout()

        fig.savefig(plots_out_dir / "stage2_posteriors.pdf")
        plt.close(fig)
