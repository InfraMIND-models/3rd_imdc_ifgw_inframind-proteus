"""Prototype of a renewal equation model for the outbreak dynamic layer.

"""
import time

import numpy as np
import numba
import pandas as pd
import scipy, scipy.stats


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

        # Crop and normalize
        # TODO: Handle zero denominators
        _view = infec_vec[:, self.gt_max_steps:]  # Skip initial given part
        normalized_infec_vec = _view / _view.sum(axis=1)[:, np.newaxis]

        # Rescale to match given total cases
        expectancy_vec = normalized_infec_vec * params_table_df["notif_total_cases"].values[:, np.newaxis]



        # Sample cases using negative binomial
        _overdisp = params_table_df["notif_nb_overdispersion"].values[:, np.newaxis]
        cases_vec = rng.negative_binomial(
            n=_overdisp,
            p=_overdisp / (_overdisp + expectancy_vec),
            size=expectancy_vec.shape,
        )


        # Prepare output data frames (from numpy arrays)
        infec_df = pd.DataFrame(
            infec_vec[:, self.gt_max_steps:], columns=simulation_t_values
        )
        infec_df.index.name = "i_simulation"
        infec_df.columns.name = "t"

        cases_vec = pd.DataFrame(
            cases_vec[:, :], columns=simulation_t_values,
        )
        cases_vec.index.name = "i_simulation"
        cases_vec.columns.name = "t"

        print(cases_vec)


        print("WATCH")
        return infec_df, cases_vec




def main():
    pass

    num_simulations = 5000
    num_time_steps = 50

    lhs_param_ranges = {
        "rt_logist_width": [5, 70],
        "rt_logist_center": [60, 200],
        "rt_logist_rmax": [1.2, 4.0],
    }

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
        "notif_total_cases": rng.normal(loc=1000, scale=0.1 * 1000, size=num_simulations)  # Sampled separately from others

    }, index=range(num_simulations))

    # Sample required parameters with LHS
    lhs_sampler = scipy.stats.qmc.LatinHypercube(d=len(lhs_param_ranges), rng=rng)
    lhs_samples = lhs_sampler.random(n=num_simulations)  # Shape: (num_simulations, num_params)

    # Scale samples from [0, 1] to the specified ranges
    l_bounds = [v[0] for v in lhs_param_ranges.values()]
    u_bounds = [v[1] for v in lhs_param_ranges.values()]
    lhs_scaled = scipy.stats.qmc.scale(lhs_samples, l_bounds, u_bounds)

    # Override values in params_table_df
    for i, col in enumerate(lhs_param_ranges.keys()):
        params_table_df[col] = lhs_scaled[:, i]

    # ---
    # Initial state of the infections vector
    initial_infec_df = pd.DataFrame(
        {
            t: np.ones(num_simulations) for t in range(0, model.gt_max_steps * model.step_dt, model.step_dt)
        }
    )
    initial_infec_df.index.name = "i_simulation"
    initial_infec_df.columns.name = "t"

    print(f"Running multiple: {num_simulations}")
    xt0 = time.time()
    results = model.run_multiple(
        num_simulations=num_simulations,
        num_time_steps=num_time_steps,
        params_table_df=params_table_df,
        initial_infec_df=initial_infec_df,
    )
    xtf = time.time()

    infec_df, cases_df = results
    cases_df: pd.DataFrame
    print(f"Execution time: {xtf - xt0:0.2e}s")

    # Preliminary visualization
    # =========
    import plotly.express as px

    def quantile_agg(q):
     return lambda sr: sr.quantile(q)

    df = pd.DataFrame(
        {
            "mean": cases_df.mean(axis=0),
            "Q02.5%": cases_df.quantile(0.025, axis=0),
            "Q97.5%": cases_df.quantile(0.975, axis=0),
        }
    )

    fig = px.line(df, y=df.columns)
    fig.show()

    print("WATCHPOINT")


if __name__ == '__main__':
    main()
