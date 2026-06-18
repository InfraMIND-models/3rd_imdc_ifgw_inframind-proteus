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
from scipy.stats import gaussian_kde

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
    use_location_ids = ["MG"]
    # use_location_ids = []  # Runs all!
    exclude_location_ids = [] # ["CE"]  # Just to test.
    use_years = list(range(2022, 2023))
    # use_years = list(range(2016, 2023))
    ncpus = 2

    # ---
    stage1_num_simulations = 2**20

    stage2_num_simulations = 2**20
    stage2_free_params = [
        "rt_logist_r_high",
        "rt_logist_start",
        # "notif_nb_overdispersion",
        # "notif_relative_scale",
    ]
    stage2_ll_temperature = 1.  # Higher values flatten the likelihood distribution
    stage2_rel_weight_cutoff = 1e-3  # Cutoff relative to maximum weight
    stage2_sampling_seed = 42  # Seed for any sampling procedure in stage 2 (e.g. KDE sampling)
    stage2_min_samples = 1000  # Minimum number of samples to keep after cutoff (overrides cutoff if not met)
    stage2_max_samples = 5000  # Maximum number of samples to keep, avoids heavy KDE calculation

    stage3_num_simulations = 2**18

    def stage2_prior(self, df: pd.DataFrame):
        """Given a data frame of parameters, calculate prior distribution weights
        to be used in stage 2 Bayesian update.
        """
        _p = pd.Series(1.0, index=df.index)

        # if "notif_relative_scale" in df.columns:
        #     _p *= _gauss(df["notif_relative_scale"], mu=5.0, sigma=3.0)
        # #
        # if "rt_logist_w_center" in df.columns:
        #     _p *= _gauss(df["rt_logist_w_center"], mu=8.0, sigma=0.2)
        #
        # if "rt_logist_dt_center" in df.columns:
        #
        #     _p *= _gauss(df["rt_logist_dt_center"], mu=60., sigma=10.0)
        #
        #     # Enforce that dt_end > dt_center
        #     if "rt_logist_dt_end" in df.columns:
        #         _p[df["rt_logist_dt_end"] <= df["rt_logist_dt_center"]] = 0.0

        return _p


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


def _rebuild_rt_curves(
        config: SimulationConfig,
        sampled_params_df: pd.DataFrame,
        simulator: RenewalSimulator
) -> pd.DataFrame:
    _config = deepcopy(config)
    _n = sampled_params_df.shape[0]
    _config.sampling.method = "given"
    _params_df = build_calibration_params_df(_n, _config.sampling, given_params=sampled_params_df)
    # t_start = float(
    #     (_config.temporal.sim_start - _config.temporal.zero_date).days
    # ) - simulator._gt_max_steps * _config.temporal.step_dt

    # Reproduce the same procedure as in the simulator
    # However it would be better to keep this directly from the simulations
    sim_start_day = float((_config.temporal.sim_start - _config.temporal.zero_date).days)
    t_start = sim_start_day - int(np.ceil(_config.gt_max / _config.temporal.step_dt))

    rt_array = simulator.rt_model.generate(
        params_df=_params_df,
        num_time_steps=_config.num_time_steps,
        step_dt=_config.temporal.step_dt,
        t_start=t_start,
    )
    # Shift time to assign dates
    _cols = _config.temporal.zero_date + pd.Timedelta(t_start, unit="D") + pd.to_timedelta(
        np.arange(rt_array.shape[1]) * _config.temporal.step_dt, unit="D")
    rt_df = pd.DataFrame(
        rt_array,
        index=sampled_params_df.index,
        columns=_cols,
    )

    return rt_df


def run_calibration_for_location_year(
        location_year_tuple,
        cfg: ProgramConfig,
        base_config_dict: dict,
        uf_table_df: pd.DataFrame,
):
    location_id, year = location_year_tuple
    print(f"run_calibration_for_location_year(({location_id}, {year}))")

    # # Load again observations, since this function will run in parallel... not worth caching
    # obserations_sr = DiseaseTimeSeriesCache(Path("data/disease/dengue_cases_uf_weekly")).get(location_id)
    # Load observations
    observations_sr = DiseaseTimeSeriesCache().get_location(location_id)

    # Stage 1: Broad parameter exploration with max. likelihood selection
    # ===============
    stage1_outputs = run_calibration_stage_1(
        location_id, year,
        cfg=cfg,
        base_config_dict=base_config_dict,
        observations_sr=observations_sr,
        uf_table_df=uf_table_df,
    )
    gc.collect()

    # Stage 2: Strict parameter explor., fixing nuisance params from stage 1
    # ==============
    stage2_outputs = run_calibration_stage_2(
        location_id, year,
        cfg=cfg,
        base_config_dict=base_config_dict,
        observations_sr=observations_sr,
        uf_table_df=uf_table_df,
        stage1_outputs=stage1_outputs,
    )
    gc.collect()

    # Stage 3: Confidence interval adjustment
    # ==============
    stage3_outputs = run_calibration_stage_3(
        location_id, year,
        cfg=cfg,
        base_config_dict=base_config_dict,
        observations_sr=observations_sr,
        uf_table_df=uf_table_df,
        stage1_outputs=stage1_outputs,
        stage2_outputs=stage2_outputs,
    )


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
) -> Stage1Outputs:
    print(f"\trun_calibration_stage_1(({location_id}, {year}))")

    # observations_sr = DiseaseTimeSeriesCache().get_location(location_id)
    config_dict = _d = deepcopy(base_config_dict)

    # --- Set configuration for this stage
    _set_config_dict_common(
        cfg, _d,
        location_id,
        year,
        uf_table_df
    )

    _d["simulation"]["num_simulations"] = cfg.stage1_num_simulations

    _scoring = _d["scoring"]
    _scoring["metrics"] = ["nb_loglikelihood"]

    # --- Manually remove overdispersion from exploration (adjusted later in stage 3)
    _sampling = _d["sampling"]
    if "notif_nb_overdispersion" in _sampling["param_ranges"]:
        del _sampling["param_ranges"]["notif_nb_overdispersion"]

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
    max_ll_params.index.name = "parameter_name"
    max_ll_params.name = "value"

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

        ax.set_ylabel("Weekly cases")
        fig.tight_layout()

        fig.savefig(dir / "stage1_max_ll_case_beam.pdf")
        plt.close(fig)

    # --- Export some stuff
    max_ll_params.to_csv(dir / "stage1_max_ll_params.csv")

    # ===

    out = Stage1Outputs(
        max_ll_params=max_ll_params,
    )

    return out


@dataclass
class Stage2Outputs:
    param_samples: pd.DataFrame
    posterior_kde: gaussian_kde


def run_calibration_stage_2(
        location_id,
        year,
        cfg: ProgramConfig,
        base_config_dict: dict,
        observations_sr: pd.Series,
        uf_table_df: pd.DataFrame,
        stage1_outputs: Stage1Outputs
):
    print(f"\trun_calibration_stage_2(({location_id}, {year}))")

    # observations_sr = DiseaseTimeSeriesCache().get_location(location_id)
    config_dict = _d = deepcopy(base_config_dict)

    # --- Set configuration for this stage
    _set_config_dict_common(
        cfg, config_dict,
        location_id,
        year,
        uf_table_df
    )

    # --- Override parameters found in stage 1
    param_ranges = config_dict["sampling"]["param_ranges"]
    _sampling = config_dict["sampling"]
    nuisance_param_names = [
        p for p in param_ranges.keys()
        if p not in cfg.stage2_free_params
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

    # # --- Manually add overdispersion to be explored
    # if "notif_nb_overdispersion" in cfg.stage2_free_params:
    #     _sampling["param_ranges"]["notif_nb_overdispersion"] = [0.1, 100.0]
    #     _sampling["param_scales"] = {
    #         "notif_nb_overdispersion": "inverse"
    #     }

    _d["simulation"]["num_simulations"] = cfg.stage2_num_simulations

    _scoring = _d["scoring"]
    _scoring["metrics"] = ["nb_loglikelihood"]

    # --- Create simulator object with modified configuration dictionary
    simulator = RenewalSimulator.from_config_dict(config_dict)
    sim_cfg = simulator.config

    # ====

    # --- Calculate scaling factor priors from pre-simulation period observations
    mean_rel_scaling, std_rel_scaling = _calc_scaling_from_presim_period(
        sim_cfg, observations_sr
    )
    sim_cfg.sampling.param_priors["notif_relative_scale"] = GammaPrior(
        mean_rel_scaling, std_rel_scaling
    )

    # --- Build auxiliary data frames for simulations
    params_df = build_calibration_params_df(sim_cfg.num_simulations, sim_cfg.sampling)
    sim_cfg.num_simulations = num_simulations = params_df.shape[
        0]  # Update in case sampling method changes number of simulations
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

    # ===

    # --- Build posterior distributions
    rng = np.random.default_rng(cfg.stage2_sampling_seed)
    # Treat the likelihood function
    df = pd.concat([params_df, results.scoring.summary], axis=1)
    tempered_ll = df["nb_loglikelihood"] / cfg.stage2_ll_temperature
    tempered_ll -= tempered_ll.max()  # Displace by max
    df["tempered_likelihood"] = np.exp(tempered_ll)

    # Apply prior
    df["prior_weight"] = cfg.stage2_prior(df)
    df["posterior_weight"] = df["tempered_likelihood"] * df["prior_weight"]

    # --- Regularize small weights
    max_weight = df["posterior_weight"].max()
    w_cutoff = max_weight * cfg.stage2_rel_weight_cutoff
    num_relevant_weights = (df["posterior_weight"] >= w_cutoff).sum()
    # Decision tree based on number of survival weights
    if num_relevant_weights < cfg.stage2_min_samples:
        # Keep min number
        df = (
            df
            .sort_values("posterior_weight")
            .iloc[-cfg.stage2_min_samples:]
        )
    elif num_relevant_weights < cfg.stage2_max_samples:
        # Keep all above cutoff
        df = (
            df[df["posterior_weight"] >= w_cutoff]
            .sort_values("posterior_weight")
        )
    else:  # num_relevant_weights >= cfg.stage2_max_samples
        # Sample from relevant weights to keep max number
        df = (
            df[df["posterior_weight"] >= w_cutoff]
            .sample(
                n=cfg.stage2_max_samples,
                weights="posterior_weight",
                random_state=rng,
            )
            .sort_values("posterior_weight")
        )
    # At this point, df contains selected samples and their weights, with a
    #    variable size between specified min and max.
    param_samples = df

    # from scipy.stats import gaussian_kde
    kde = gaussian_kde(
        param_samples[cfg.stage2_free_params].T.values,
        weights=param_samples["posterior_weight"]
    )


    # =====
    # --- Visualize median trajectories against data
    rc = deepcopy(plt.rcParams)
    # Sample pre-simulated trajectories with posteriors as weights
    sampled_params_df: pd.DataFrame = params_df.sample(
        n=500, weights=df["posterior_weight"], replace=True, random_state=123,
    )
    sampled_beams = results.case_beam_df.loc[
        results.case_beam_df.index.get_level_values("i_simulation").isin(sampled_params_df.index)
    ]
    # Reproduction number from the same sampled trajectories
    # Rebuild the full params df but only with the best sampled parameters
    rt_df = _rebuild_rt_curves(results.config, sampled_params_df, simulator)

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
        ax = axes[1]
        ax.plot(rt_df.T, color="darkslateblue", alpha=0.2)
        ax.plot(rt_df.columns, np.ones(rt_df.shape[1]), "k--", label="R=1")
        ax.set_ylabel("Reproduction number")
        ax.set_xlim(*axes[0].get_xlim())  # Align x-axis with case trajectories

        fig.tight_layout()
        dir = Path(f".local/calibrate_multistage/{location_id}_{year}")
        dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(dir / "stage2_median_trajectories.pdf")
        plt.close(fig)



    # --- Visualize posteriors
    num_params = len(cfg.stage2_free_params)
    param_ranges = results.config.sampling.param_ranges
    rc = deepcopy(plt.rcParams)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows=num_params, ncols=1,
            figsize=(8, num_params * 3)
        )

        for i, param_name in enumerate(cfg.stage2_free_params):
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

        dir = Path(f".local/calibrate_multistage/{location_id}_{year}")
        dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(dir / "stage2_posteriors.pdf")
        plt.close(fig)


    out = Stage2Outputs(
        param_samples=df,
        posterior_kde=kde
    )

    return out


@dataclass
class Stage3Outputs:
    pass


def run_calibration_stage_3(
        location_id,
        year,
        cfg: ProgramConfig,
        base_config_dict: dict,
        observations_sr: pd.Series,
        uf_table_df: pd.DataFrame,
        stage1_outputs: Stage1Outputs,
        stage2_outputs: Stage2Outputs,
):
    print(f"\trun_calibration_stage_3(({location_id}, {year}))")

    config_dict = _d = deepcopy(base_config_dict)

    # --- Set configuration for this stage
    _set_config_dict_common(
        cfg, config_dict,
        location_id,
        year,
        uf_table_df
    )

    # --- Override parameters found in stage 1
    param_ranges: dict = config_dict["sampling"]["param_ranges"]
    _sampling = config_dict["sampling"]
    nuisance_param_names = [
        p for p in param_ranges.keys()
        if p not in cfg.stage2_free_params
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

    # Manually clear exploration parameters, we'll sample them separately
    overdisp_range = param_ranges.get("notif_nb_overdispersion", [0.1, 100.0])
    param_ranges.clear()


    _d["simulation"]["num_simulations"] = cfg.stage3_num_simulations
    _scoring = _d["scoring"]
    _scoring["metrics"] = ["nb_loglikelihood", "coverages"]

    # --- Create simulator object with modified configuration dictionary
    simulator = RenewalSimulator.from_config_dict(config_dict)
    sim_cfg = simulator.config

    # ====

    # --- Build auxiliary data frames for simulations
    params_df = build_calibration_params_df(sim_cfg.num_simulations, sim_cfg.sampling)
    sim_cfg.num_simulations = num_simulations = params_df.shape[0]  # Update in case sampling method changes number of simulations
    initial_infec_df = build_initial_infec_df(
        sim_cfg.num_simulations,
        simulator._gt_max_steps,
        sim_cfg.temporal.step_dt,
        sim_cfg.initial_infections
    )

    # --- Override params_df with the manually sampled stuff
    # Random independent sampling: KDE x Overdispersion
    n_samples = params_df.shape[0]
    rng = np.random.default_rng(321)  # TODO: create a seed parameter
    kde_param_samples = stage2_outputs.posterior_kde.resample(n_samples, seed=rng).T
    # Sample overdispersion inversely
    overdisp_param_samples = 1. / (
        rng.uniform(1. / overdisp_range[1], 1. / overdisp_range[0], size=n_samples)
    )
    params_df[cfg.stage2_free_params] = kde_param_samples
    params_df["notif_nb_overdispersion"] = overdisp_param_samples

    # ====
    # --- RUN
    results = simulator.run_sequential_chunks(
        params_df=params_df,
        initial_infec_df=initial_infec_df,
        observations_sr=observations_sr,
    )

    df = pd.concat([params_df, results.scoring.summary], axis=1)

    # --- Test: Maximum NB likelihood
    # TODO: Formalize parameter selection / posterior calculation
    i_max = df["coverage_loglikelihood"].idxmax()
    best_sim = results.case_beam_df.xs(i_max, level="i_simulation")

    fig, ax = plt.subplots()
    _obs = observations_sr.loc[sim_cfg.temporal.sim_start:sim_cfg.temporal.calibration_end]
    ax.plot(_obs.index, _obs.values, "ks", label="Observations")
    ax.plot(best_sim.loc[0.5].T, color="darkslateblue", alpha=0.7, label="Best sim")
    ax.fill_between(
        best_sim.columns,
        best_sim.loc[0.25], best_sim.loc[0.75],
        color="darkslateblue", alpha=0.3, label="50% CI"
    )
    ax.fill_between(
        best_sim.columns,
        best_sim.loc[0.025], best_sim.loc[0.975],
        color="darkslateblue", alpha=0.3, label="95% CI"
    )

    fig.tight_layout()
    dir = Path(f".local/calibrate_multistage/{location_id}_{year}")
    dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(dir / "stage3_best_sim.pdf")
    plt.close(fig)



if __name__ == "__main__":
    main()
