"""Prototype of a renewal equation model for the outbreak dynamic layer.

"""
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numba
import pandas as pd
import scipy, scipy.stats
from numpy.random import Generator


def sample_lhs(
        lhs_param_ranges: dict[str, list[int] | list[float]],
        num_simulations: int,
        rng: Generator | Generator
) -> pd.DataFrame:

    # Sample required parameters with LHS
    lhs_sampler = scipy.stats.qmc.LatinHypercube(d=len(lhs_param_ranges), rng=rng)
    lhs_samples = lhs_sampler.random(n=num_simulations)  # Shape: (num_simulations, num_params)

    # Scale samples from [0, 1] to the specified ranges
    l_bounds = [v[0] for v in lhs_param_ranges.values()]
    u_bounds = [v[1] for v in lhs_param_ranges.values()]
    lhs_scaled = scipy.stats.qmc.scale(lhs_samples, l_bounds, u_bounds)

    lhs_scaled_df = pd.DataFrame(
        lhs_scaled,
        columns=list(lhs_param_ranges.keys()),
    )

    return lhs_scaled_df



def create_logistic_rt(
        params_table_df: pd.DataFrame,
        num_simulations: int,
        num_time_steps: int,
        step_dt: int,
) -> np.ndarray:
    """
    Create a 2D array of logistic reproduction number time series.

    Parameters
    ----------
    params_table_df : pd.DataFrame
        Parameter table with columns: rt_logist_start, rt_logist_center,
        rt_logist_width, rt_logist_rmin, rt_logist_rmax
    num_simulations : int
        Number of simulations (rows in output)
    num_time_steps : int
        Number of time steps (columns in output)
    step_dt : int
        Time step duration in days

    Returns
    -------
    np.ndarray
        2D array of shape (num_simulations, num_time_steps) with logistic curves.
        Values are roff before rt_logist_start, then follow negative-amplitude
        logistic function: rmin + (rmax - rmin) / (1 + exp((center - t) / width))
    """
    # Create time grid: columns are time steps (in days)
    time_grid = np.arange(0, num_time_steps * step_dt, step_dt)  # Shape: (num_time_steps,)

    # Extract logistic parameters from table
    rt_roff = params_table_df["rt_logist_roff"].values
    rt_start = params_table_df["rt_logist_start"].values
    rt_center = params_table_df["rt_logist_center"].values
    rt_width = params_table_df["rt_logist_width"].values
    rt_rmin = params_table_df["rt_logist_rmin"].values
    rt_rmax = params_table_df["rt_logist_rmax"].values

    # Reshape for broadcasting: (num_simulations, 1) and (num_time_steps,)
    rt_roff_r = rt_roff[:, np.newaxis]
    rt_start_r = rt_start[:, np.newaxis]
    rt_center_r = rt_center[:, np.newaxis]
    rt_width_r = rt_width[:, np.newaxis]
    rt_rmin_r = np.atleast_1d(rt_rmin)[:, np.newaxis]
    rt_rmax_r = np.atleast_1d(rt_rmax)[:, np.newaxis]

    # Only compute for t >= start
    t_mask = time_grid >= rt_start_r  # Shape: (num_simulations, num_time_steps)

    # Logistic function with negative amplitude (inverted sigmoid)
    # rmin + (rmax - rmin) / (1 + exp((center - t) / width))
    exponent = (rt_center_r - time_grid) / rt_width_r
    logistic_vals = rt_rmax_r - (rt_rmax_r - rt_rmin_r) / (1 + np.exp(exponent))

    # Apply mask to set values to zero before start
    rt_vec = np.where(t_mask, logistic_vals, rt_roff_r)

    return rt_vec


def nbinom_ppf_cf(q, n, p, continuity=True):
    """
    Continuous Cornish-Fisher approximation to the Negative Binomial PPF.

    References in: https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion

    Parameters
    ----------
    q : float
        Quantile in (0, 1).

    n : array_like
        Number of successes parameter (> 0).

    p : array_like
        Success probability parameter in (0, 1).

    continuity : bool, default=True
        Whether to apply a +0.5 continuity correction
        before flooring to integers.

    Returns
    -------
    k : ndarray
        Approximate quantiles (non-integer) of the Negative Binomial distribution.

    Notes
    -----
    Assumes the SciPy parameterization:

        X ~ nbinom(n, p)

    where X counts failures before n successes.

    Uses the Cornish-Fisher expansion:

        x ≈ μ + σ [ z + (γ1 / 6)(z² - 1) ]

    with

        μ  = n(1-p)/p
        σ² = n(1-p)/p²
        γ1 = (2-p)/sqrt(n(1-p))
    """
    from scipy.stats import norm

    n = np.asarray(n, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    z = norm.ppf(q)
    z2 = z * z
    z3 = z2 * z

    # Mean and standard deviation
    mu = n * (1.0 - p) / p
    sigma = np.sqrt(n * (1.0 - p)) / p

    # Skewness
    gamma1 = (2.0 - p) / np.sqrt(n * (1.0 - p))
    gamma2 = (p * p - 6.0 * p + 6.0) / (n * (1.0 - p))  # Third order term

    # Cornish-Fisher correction
    # cf = z + (gamma1 / 6.0) * (z2 - 1.0)  # Second order
    cf = (  # Third order
        z
        + (gamma1 / 6.0) * (z2 - 1.0)
        + (gamma2 / 24.0) * (z3 - 3.0 * z)
        - (gamma1 * gamma1 / 36.0) * (2.0 * z3 - 5.0 * z)
    )

    x = mu + sigma * cf

    if continuity:
        # x = np.floor(x + 0.5)
        x += 0.5
    else:
        pass
        # x = np.floor(x)

    # Negative binomial support starts at 0
    return np.maximum(x, 0)#.astype(np.int64)


def wis_score_vectorized(
        simulations_df: pd.DataFrame,
        observations_sr: pd.Series,
        alphas = None,
        weights = None,
        weight_of_median = 0.5,
):
    """Compute the Weighted Interval Score (WIS) for a set of quantiles
    and observations.
    Operations are vectorized to handle large sets of simulations at once.

    simulations_df: DataFrame with MultiIndex (quantile, i_simulation) and columns as time steps
    observations_sr: Series with index as time steps and values as observed cases

    Number of time steps must match between simulations_df and observations_sr.
    """
    # ============ Preamble
    assert simulations_df.columns.equals(observations_sr.index), \
        "Time steps in simulations and observations must match"

    available_q = simulations_df.index.get_level_values("quantile").unique().values
    available_q.sort()

    assert 0.5 in available_q, "Median (0.5 quantile) must be available for WIS calculation"

    if alphas is None:
        # Infer from available quantiles
        use_q = [q for q in available_q if q < 0.5]
        alphas = np.array(
            [2 * q for q in use_q]
        )

    if weights is None:
        weights = alphas / 2.  # Default WIS weights

    # TODO: Assert simmetry of quantiles to be used

    # ========= Calculation

    # Precompute the expanded observation vector for broadcasting
    obs_vec = observations_sr.values[np.newaxis, :]  # Shape: (1, num_time_steps)

    wis_component_list = list()
    for alpha in alphas:
        q_low = alpha / 2
        q_high = 1 - alpha / 2

        pred_low_vec = simulations_df.xs(q_low, level="quantile").values
        pred_high_vec = simulations_df.xs(q_high, level="quantile").values
        # Shape: (num_simulations, num_time_steps)

        high_pred_penalty = np.maximum(0, obs_vec - pred_high_vec)
        low_pred_penalty = np.maximum(0, pred_low_vec - obs_vec)

        wis_sharpness = pred_high_vec - pred_low_vec  # WIS sharpness
        wis_calibration = (2 / alpha) * (high_pred_penalty + low_pred_penalty)
        wis_component = wis_sharpness + wis_calibration

        wis_component_list.append(wis_component)

    # Separate calculation for median
    pred_median = simulations_df.xs(0.5, level="quantile").values
    wis_median_component = np.abs(pred_median - obs_vec)
    # Shape: (num_simulations, num_time_steps)

    # Aggregate WIS components accross quantiles

    # Start with the median
    wis = wis_median_component * weight_of_median
    # Shape: (num_simulations, num_time_steps)

    for i, alpha in enumerate(alphas):
        wis += wis_component_list[i] * weights[i]

    # Final normalization (number of prediction intervals)
    wis /= (len(alphas) + 0.5)

    return wis



class ProtoDynModel:
    """Prototype dynamic model"""

    gt_max = 49  # In days
    step_dt = 7  # Length of step in days - 7 for weekly


    def __init__(self):

        # === Dependent parameters
        # Generation time
        self.gt_max_steps = int(np.ceil(self.gt_max / self.step_dt))



    # === Generation time vector
    def run_multiple(
            self,
            num_simulations: int,
            num_time_steps: int,
            params_table_df: pd.DataFrame,
            initial_infec_df: pd.DataFrame,
            notif_rng_seed: int = 0,
            case_beam_quantiles = None
    ):
        """
        Signature of params_table:
        - Each row is one simulation
        - Each colum is one parameter

        Signature of initial_infec_df
        - Each row is one simulation
        - Each column is one time step
        - Must have gt_max_steps time steps
        """
        # Check preconditions
        assert initial_infec_df.shape[0] == num_simulations, "initial_infec_df must have num_simulations rows"
        assert initial_infec_df.shape[1] == self.gt_max_steps

        if case_beam_quantiles is None:
            case_beam_quantiles = np.array([
                0.025, 0.25, 0.5, 0.75, 0.975
            ])

        # ----------
        simulation_i_steps = np.arange(
            self.gt_max_steps,
            self.gt_max_steps + num_time_steps
        )
        simulation_t_values = self.step_dt * simulation_i_steps

        # Create the generation time
        # NOTE: Goes one step above the maximum
        # -()- Gamma
        gt_vals = np.arange(
            0, (self.gt_max_steps + 1) * self.step_dt, self.step_dt
        )
        gt_vals_reshaped = gt_vals[np.newaxis, :]  # Shape: (1, len(gt_vals))
        a_vals = params_table_df["gt_gamma_shape"].values[:, np.newaxis]  # Shape: (num_simulations, 1)
        scale_vals = params_table_df["gt_gamma_scale"].values[:, np.newaxis]  # Shape: (num_simulations, 1)

        gamma_cdf_samples = scipy.stats.gamma.cdf(
            gt_vals_reshaped,
            a=a_vals,
            scale=scale_vals,
        )

        gt_pmf = np.diff(gamma_cdf_samples, axis=1)
        gt_pmf_reverse = gt_pmf[:, ::-1]
        # Result shape: (num_simulations, gt_max_steps)

        # Create the pre-fixed reproduction number curve

        # -()- Logistic
        rt_vec = create_logistic_rt(
            params_table_df=params_table_df,
            num_simulations=num_simulations,
            num_time_steps=(self.gt_max_steps + num_time_steps),
            step_dt=self.step_dt,
        )
        # Result shape: (num_simulations, gt_max_steps + num_time_steps)


        # ---

        # Create the full vector of infections
        infec_vec = np.concatenate(
            [
                initial_infec_df.to_numpy(),
                np.zeros((num_simulations, num_time_steps), dtype=float),
            ], axis=1
        )
        # Result shape: (num_simulations, gt_max_steps + num_time_steps)

        # Abstract infections model (hidden process)
        # ==================

        # Time loop, vectorized for multiple simulations
        for i_step in simulation_i_steps:
            i_step_start = i_step - self.gt_max_steps

            # Calculate next step by convolving the three vectors
            # This step applies the renewal equation
            infec_vec[:, i_step] = (
                np.sum(
                    rt_vec[:, i_step_start:i_step]
                    * infec_vec[:, i_step_start:i_step]
                    * gt_pmf_reverse,
                    axis=1
                )
            )

        print(infec_vec)


        # Infection to notifications model (observation process)
        # =================
        rng = np.random.default_rng(notif_rng_seed)

        # Crop the initial part (artificially given)
        _view = infec_vec[:, self.gt_max_steps:]  # Skip initial given part

        # # -()- Normalize by infection attack rate (curve area)
        # TODO: Handle zero denominators
        # normalized_infec_vec = _view / _view.sum(axis=1)[:, np.newaxis]
        # # Rescale to match given total cases
        # expectancy_vec = normalized_infec_vec * params_table_df["notif_total_cases"].values[:, np.newaxis]

        # -()- Apply given scaling factor
        expectancy_vec = _view * params_table_df["notif_scaling_factor"].values[:, np.newaxis]


        print("WATCHPOINT")

        # --- Sample cases using negative binomial
        _overdisp = params_table_df["notif_nb_overdispersion"].values[:, np.newaxis]
        cases_vec = rng.negative_binomial(
            n=_overdisp,
            p=_overdisp / (_overdisp + expectancy_vec),
            size=expectancy_vec.shape,
        )
        # Signature: (num_simulations, num_time_steps)

        # --- Calculate case beam quantiles for each simulaiton with negative binomial
        # OBS: TOooo slow for large num_simulations
        # _overdisp = params_table_df["notif_nb_overdispersion"].values[:, np.newaxis]
        case_beam_df_list = list()
        for q in case_beam_quantiles:
            # # -()- Exact with scipy (slow if large)
            # q_vec = scipy.stats.nbinom.ppf(
            #     q=q,
            #     n=_overdisp,
            #     p=_overdisp / (_overdisp + expectancy_vec),
            # )

            # -()- Approximate with Cornish-Fisher (fast)
            q_vec = nbinom_ppf_cf(
                q=q,
                n=_overdisp,
                p=_overdisp / (_overdisp + expectancy_vec), continuity=False

            )

            # - Compare the two
            # _diff = q_vec_cf - q_vec
            # _rel_diff = _diff / expectancy_vec
            #
            # import matplotlib.pyplot as plt
            # reldiff_df = pd.DataFrame(_rel_diff)
            # fig, ax = plt.subplots()
            #
            # ax.plot(_diff.T)
            # fig.show()

            # Signature of q_vec: (num_simulations, num_time_steps)

            case_beam_df_list.append(pd.DataFrame(q_vec))

        case_beam_df = pd.concat(case_beam_df_list, keys=case_beam_quantiles, names=["quantile", "i_simulation"])
        # Signature: df.loc[(quantile, i_simulation), t] = quantile of the case "beam"

        print(case_beam_df)

        # Prepare output data frames (from numpy arrays)
        infec_df = pd.DataFrame(
            infec_vec[:, self.gt_max_steps:], columns=simulation_t_values
        )
        infec_df.index.name = "i_simulation"
        infec_df.columns.name = "t"

        cases_df = pd.DataFrame(
            cases_vec[:, :], columns=simulation_t_values,
        )
        cases_df.index.name = "i_simulation"
        cases_df.columns.name = "t"

        print(cases_df)


        # print("WATCH")
        return infec_df, cases_df, case_beam_df


def main():

    num_simulations = 1000000
    num_time_steps = 50

    # --- Select location & start date
    # Obs: Target files preliminarily generated in `explore_mosqlimate_data.ipynb`
    # uf = "AP"
    uf = "DF"
    # uf = "RO"
    # uf = "RR"
    # uf = "SP"
    # date_zero = pd.Timestamp("2015-10-05")
    # date_zero = pd.Timestamp("2019-10-06")
    # date_zero = pd.Timestamp("2021-10-04")
    # date_zero = pd.Timestamp("2022-10-03")
    date_zero = pd.Timestamp("2023-10-02")

    tgt_data_df = pd.read_csv(
        f".local/dengue/cases_tseries_{uf}.csv",
        parse_dates=["date"]
    )


    # # To be implemented: Different scales for sampling
    # sampling_scale_dict = defaultdict(
    #     lambda: "linear",
    #     notif_scaling_factor="log10",
    # )

    # ======

    model = ProtoDynModel()
    rng = np.random.default_rng(seed=123)  # Local RNG for sampling


    params_table_df = pd.DataFrame({
        # Generation time
        "gt_gamma_shape":10.0,
        "gt_gamma_scale": 1.8,  # Expectancy = product

        # R(t) stepped logistic function
        "rt_logist_width": 13.,  # In days
        "rt_logist_center": 100.,  # In days
        "rt_logist_roff": 1.0,  # Off-season R value (before start)
        "rt_logist_start": 50.,
        "rt_logist_rmin": 0.2,  # Post-outbreak baseline
        "rt_logist_rmax": 2.5,  # Essentially "R0"

        # Infection-to-notification model
        "notif_nb_overdispersion": 10.,
        "notif_total_cases": rng.normal(loc=1000, scale=0.1 * 1000, size=num_simulations),  # Sampled separately from others
        "notif_scaling_factor": 1.0,  # Only applied if using external factor

    }, index=range(num_simulations))

    # Parameters to be explored
    lhs_param_ranges = {
        "rt_logist_width": [1, 80],
        "rt_logist_center": [10, 280],
        "rt_logist_rmax": [1.1, 4.0],
        "notif_nb_overdispersion": [0.1, 100],
        "notif_scaling_factor": [10, 3000],
    }

    # Override sampled values in params_table_df
    lhs_scaled_df = sample_lhs(lhs_param_ranges, num_simulations, rng)
    for col in lhs_scaled_df.columns:
        params_table_df[col] = lhs_scaled_df[col]

    # ---
    # Initial state of the infections vector
    initial_infec_df = pd.DataFrame(
        {
            t: np.ones(num_simulations) for t in range(0, model.gt_max_steps * model.step_dt, model.step_dt)
        }
    )
    initial_infec_df.index.name = "i_simulation"
    initial_infec_df.columns.name = "t"

    # Run the model once
    # =================
    print(f"Running multiple: {num_simulations}")
    xt0 = time.time()
    results = model.run_multiple(
        num_simulations=num_simulations,
        num_time_steps=num_time_steps,
        params_table_df=params_table_df,
        initial_infec_df=initial_infec_df,
    )
    xtf = time.time()

    infec_df, cases_df, case_beam_df = results
    cases_df: pd.DataFrame
    print(f"Execution time: {xtf - xt0:0.2e}s")


    # Evaluate all simulations (case beams)
    # ============
    # Crop observations to the same time range
    sr = tgt_data_df.set_index("date")["casos"]
    date_start = date_zero
    date_end = date_zero + pd.Timedelta(weeks=num_time_steps)
    sr = sr.loc[date_start:date_end - pd.Timedelta(days=1)]  # Inclusive end, so subtract one day
    observations_sr = sr.reset_index(drop=True)

    assert sr.shape[0] == case_beam_df.shape[1]  # Check number of time steps

    # Calculate WIS for all simulations
    wis_array = wis_score_vectorized(
        simulations_df=case_beam_df,
        observations_sr=observations_sr
    )
    # Shape: (num_simulations, num_time_steps)
    wis_total_array = wis_array.sum(axis=1)  # Total WIS for each simulation
    # Shape: (num_simulations,)





    # Preliminary visualization
    # =========
    import plotly.express as px
    import matplotlib.pyplot as plt
    plt.rcParams["patch.linewidth"] = 0

    def quantile_agg(q):
        return lambda sr: sr.quantile(q)

    # --- Plot the exact best simulation and the data
    best_wis_idx = np.argmin(wis_total_array)
    best_simulation_cases_sr = cases_df.iloc[best_wis_idx]
    df = case_beam_df.xs(best_wis_idx, level="i_simulation")

    pred_df = df.T
    # Shape: (num_time_steps, num_quantiles)

    #
    fig, ax = plt.subplots()
    ax.plot(
        pred_df.index, pred_df[0.5], "--", label="Predicted median", color="midnightblue"
    )
    for iqr in [0.5, 0.95]:
        q_low = round((1 - iqr) / 2, 6)
        q_high = round(1 - q_low, 6)
        ax.fill_between(
            pred_df.index,
            pred_df[q_low],
            pred_df[q_high],
            alpha=0.3,
            label=f"{int(iqr*100)}% prediction interval",
            color="midnightblue"
        )
    ax.plot(observations_sr, "o", color="k", label="Observed")

    ax.legend()

    fpath = Path(f".local/prototype_renewal/best_simulation_{uf}_{date_start.date().isoformat()}.pdf")
    fpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(fpath)
    fig.show()


    print("WATCHPOINT")

    # # --- Plot mean and quantile of case trajectories
    # df = pd.DataFrame(
    #     {
    #         "mean": cases_df.mean(axis=0),
    #         "Q02.5%": cases_df.quantile(0.025, axis=0),
    #         "Q97.5%": cases_df.quantile(0.975, axis=0),
    #     }
    # )
    #
    # fig = px.line(df, y=df.columns)
    #
    # _fpath = Path(".local/prototype_renewal_tseries.html")
    # _fpath.parent.mkdir(exist_ok=True, parents=True)
    # fig.write_html(_fpath)

    # --- Plot case trajectories for a random subset of simulations
    num_traces = min(500, num_simulations)
    idx = rng.choice(num_simulations, size=num_traces, replace=False)
    cases_sample_df = cases_df.iloc[idx]

    fig = px.line(
        cases_sample_df.T.reset_index(),
        x="t",
        y=cases_sample_df.index,
        # title=f"Sample of {num_traces} case trajectories",
        # labels={"t": "Time (days)", "value": "Cases", "variable": "Simulation"},
    )
    _fpath = Path(".local/prototype_renewal_trajectories.html")
    _fpath.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(_fpath)

    # import plotly.io as pio
    # print(pio.renderers)  # Check ways to render a plotly figure

    # fig.show()

    # print("WATCHPOINT")



if __name__ == '__main__':
    main()
