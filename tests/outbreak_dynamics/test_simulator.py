"""Unit tests for outbreak_dynamics.simulator.

Structure
---------
TestRenewalLoop
    Tests for _run_renewal_loop in isolation (static method, pure numpy).
    Uses analytically-verifiable setups (delta GT, flat R(t), etc.).

TestObservationModel
    Tests for _apply_observation_model (shapes, dtype, determinism, edge cases).

TestRun
    Integration tests for RenewalSimulator.run() — shapes, validation errors,
    column timestamps, calibration vs. projection mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from inframind_proteus.outbreak_dynamics.generation_time import ConstantGammaGT
from inframind_proteus.outbreak_dynamics.rt_models import LogisticRT
from inframind_proteus.outbreak_dynamics.simulator import (
    LocationConfig,
    RenewalSimulator,
    SimulationConfig,
    SimulationOutput,
    TemporalConfig,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_delta_gt_pmf(gt_max_steps: int, num_time_steps: int) -> np.ndarray:
    """GT PMF that puts all weight on lag-1 (most recent step).

    In the reversed-lag convention, index -1 = lag 1, so we set the last
    column to 1 and everything else to 0.
    """
    gt = np.zeros((num_time_steps, gt_max_steps))
    gt[:, -1] = 1.0
    return gt


def make_config(
    num_sim: int = 3,
    num_steps: int = 4,
    step_dt: int = 7,
    gt_max: int = 7,
    mode: str = "projection",
    zero_date: str = "2024-01-01",
    sim_start: str = "2024-01-01",
    rng_seed: int = 42,
    case_beam_quantiles: list[float] | None = None,
    calibration_start: str | None = None,
    calibration_end: str | None = None,
) -> SimulationConfig:
    if case_beam_quantiles is None:
        case_beam_quantiles = [0.025, 0.5, 0.975]
    return SimulationConfig(
        mode=mode,
        num_simulations=num_sim,
        num_time_steps=num_steps,
        gt_max=gt_max,
        temporal=TemporalConfig(
            zero_date=pd.Timestamp(zero_date),
            sim_start=pd.Timestamp(sim_start),
            step_dt=step_dt,
            calibration_start=pd.Timestamp(calibration_start) if calibration_start else None,
            calibration_end=pd.Timestamp(calibration_end) if calibration_end else None,
        ),
        case_beam_quantiles=case_beam_quantiles,
        rng_seed=rng_seed,
    )


def make_params_df(
    n: int,
    roff: float = 1.0,
    start: float = -1e6,   # far before t=0 so window is always active
    end: float = 1e6,
    center: float = 5e5,
    width: float = 14.0,
    rmin: float = 0.9,
    rmax: float = 1.1,
    overdisp: float = 100.0,  # high → nearly Poisson, less randomness in tests
    scale: float = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rt_logist_roff":   [roff]    * n,
            "rt_logist_start":  [start]   * n,
            "rt_logist_end":    [end]     * n,
            "rt_logist_center": [center]  * n,
            "rt_logist_width":  [width]   * n,
            "rt_logist_rmin":   [rmin]    * n,
            "rt_logist_rmax":   [rmax]    * n,
            "notif_nb_overdispersion": [overdisp] * n,
            "notif_scaling_factor":    [scale]    * n,
        }
    )


def make_simulator(
    num_sim: int = 3,
    num_steps: int = 4,
    step_dt: int = 7,
    gt_max: int = 7,
    mode: str = "projection",
    rng_seed: int = 42,
) -> RenewalSimulator:
    """Minimal simulator using LogisticRT + ConstantGammaGT."""
    cfg = make_config(
        num_sim=num_sim,
        num_steps=num_steps,
        step_dt=step_dt,
        gt_max=gt_max,
        mode=mode,
        rng_seed=rng_seed,
    )
    return RenewalSimulator(
        rt_model=LogisticRT(),
        gt_model=ConstantGammaGT(shape=5.0, scale=2.0),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# TestRenewalLoop
# ---------------------------------------------------------------------------

class TestRenewalLoop:
    """Tests for RenewalSimulator._run_renewal_loop (static, pure numpy)."""

    def test_zero_rt_gives_zero_simulation_steps(self):
        """R(t) = 0 everywhere → all computed steps are 0."""
        gt_steps, num_steps, num_sim = 3, 5, 4
        gt_pmf = make_delta_gt_pmf(gt_steps, num_steps)

        infec_vec = np.ones((num_sim, gt_steps + num_steps))
        infec_vec[:, gt_steps:] = 0.0  # only warm-up pre-filled
        rt_vec = np.zeros_like(infec_vec)

        result = RenewalSimulator._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, num_steps
        )
        assert_allclose(result[:, gt_steps:], 0.0)

    def test_warm_up_columns_unchanged(self):
        """The first gt_max_steps columns must never be modified."""
        gt_steps, num_steps, num_sim = 3, 5, 4
        warmup_values = np.random.default_rng(0).random((num_sim, gt_steps))

        infec_vec = np.zeros((num_sim, gt_steps + num_steps))
        infec_vec[:, :gt_steps] = warmup_values

        rt_vec = np.ones_like(infec_vec)
        gt_pmf = make_delta_gt_pmf(gt_steps, num_steps)

        RenewalSimulator._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, num_steps
        )
        assert_allclose(infec_vec[:, :gt_steps], warmup_values)

    def test_in_place_and_return_same_object(self):
        """The function modifies infec_vec in-place and returns the same array."""
        gt_steps, num_steps, num_sim = 2, 3, 2
        infec_vec = np.ones((num_sim, gt_steps + num_steps))
        infec_vec[:, gt_steps:] = 0.0
        rt_vec = np.ones_like(infec_vec)
        gt_pmf = make_delta_gt_pmf(gt_steps, num_steps)

        returned = RenewalSimulator._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, num_steps
        )
        assert returned is infec_vec

    def test_delta_gt_lag1_rt1_constant_trajectory(self):
        """Delta GT at lag 1, R(t)=1, uniform warm-up=c → all steps equal c.

        Renewal: I(t) = 1 * I(t-1) * 1 = I(t-1)
        """
        gt_steps = 3
        num_steps = 6
        num_sim = 4
        c = 5.0

        gt_pmf = make_delta_gt_pmf(gt_steps, num_steps)  # all weight on lag 1
        infec_vec = np.full((num_sim, gt_steps + num_steps), c)
        infec_vec[:, gt_steps:] = 0.0
        rt_vec = np.ones_like(infec_vec)

        result = RenewalSimulator._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, num_steps
        )
        assert_allclose(result[:, gt_steps:], c, rtol=1e-12)

    def test_delta_gt_lag1_rt2_exponential_growth(self):
        """Delta GT at lag 1, R(t)=2, warm-up last column=1 → doubling each step.

        Renewal: I(t) = 2 * I(t-1) * 1 → geometric sequence 2^k.
        """
        gt_steps = 3
        num_steps = 5
        num_sim = 2

        gt_pmf = make_delta_gt_pmf(gt_steps, num_steps)
        # Warm-up: last column = 1, rest = 0 (so only lag-1 matters)
        infec_vec = np.zeros((num_sim, gt_steps + num_steps))
        infec_vec[:, gt_steps - 1] = 1.0  # I at lag 1 of the first sim step
        rt_vec = np.full_like(infec_vec, 2.0)

        result = RenewalSimulator._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, num_steps
        )
        expected = 2.0 ** np.arange(1, num_steps + 1)  # 2, 4, 8, 16, 32
        for row in result:
            assert_allclose(row[gt_steps:], expected, rtol=1e-12)

    def test_uniform_gt_mass_conserved(self):
        """Uniform GT + R=1: total probability mass conserved each step."""
        gt_steps = 4
        num_steps = 6
        num_sim = 3

        # Uniform GT: each lag gets equal weight
        gt_pmf = np.full((num_steps, gt_steps), 1.0 / gt_steps)

        # Warm-up all ones
        infec_vec = np.zeros((num_sim, gt_steps + num_steps))
        infec_vec[:, :gt_steps] = 1.0
        rt_vec = np.ones_like(infec_vec)

        result = RenewalSimulator._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, num_steps
        )
        # With uniform GT and R=1, infec at each step = mean of last gt_steps = 1
        assert_allclose(result[:, gt_steps:], 1.0, rtol=1e-12)

    def test_multiple_simulations_independent(self):
        """Two simulations with different R(t) produce independent trajectories."""
        gt_steps = 2
        num_steps = 4
        num_sim = 2

        gt_pmf = make_delta_gt_pmf(gt_steps, num_steps)
        infec_vec = np.zeros((num_sim, gt_steps + num_steps))
        infec_vec[:, gt_steps - 1] = 1.0  # lag-1 seed

        rt_vec = np.zeros_like(infec_vec)
        rt_vec[0, :] = 2.0   # sim 0: R=2 → exponential growth
        rt_vec[1, :] = 0.5   # sim 1: R=0.5 → exponential decay

        result = RenewalSimulator._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, num_steps
        )
        # Sim 0 values should be strictly larger than sim 1 at every step
        assert np.all(result[0, gt_steps:] > result[1, gt_steps:])


# ---------------------------------------------------------------------------
# TestObservationModel
# ---------------------------------------------------------------------------

class TestObservationModel:
    """Tests for RenewalSimulator._apply_observation_model."""

    @pytest.fixture
    def sim(self):
        return make_simulator(num_sim=4, num_steps=5, rng_seed=0)

    @pytest.fixture
    def params(self, sim):
        return make_params_df(sim.config.num_simulations)

    def test_cases_vec_shape(self, sim, params):
        infec = np.ones((sim.config.num_simulations, sim.config.num_time_steps))
        rng = np.random.default_rng(0)
        cases_vec, _ = sim._apply_observation_model(infec, params, rng)
        assert cases_vec.shape == (sim.config.num_simulations, sim.config.num_time_steps)

    def test_cases_vec_dtype_integer(self, sim, params):
        infec = np.ones((sim.config.num_simulations, sim.config.num_time_steps))
        rng = np.random.default_rng(0)
        cases_vec, _ = sim._apply_observation_model(infec, params, rng)
        assert np.issubdtype(cases_vec.dtype, np.integer)

    def test_case_beam_multiindex_levels(self, sim, params):
        infec = np.ones((sim.config.num_simulations, sim.config.num_time_steps))
        rng = np.random.default_rng(0)
        _, case_beam_df = sim._apply_observation_model(infec, params, rng)
        assert case_beam_df.index.names == ["quantile", "i_simulation"]

    def test_case_beam_quantile_levels(self, sim, params):
        infec = np.ones((sim.config.num_simulations, sim.config.num_time_steps))
        rng = np.random.default_rng(0)
        _, case_beam_df = sim._apply_observation_model(infec, params, rng)
        beam_quantiles = sorted(
            case_beam_df.index.get_level_values("quantile").unique()
        )
        assert beam_quantiles == sorted(sim.config.case_beam_quantiles)

    def test_zero_infections_give_zero_cases(self, sim, params):
        """With zero infections, expected cases = 0 → NB always draws 0."""
        infec = np.zeros((sim.config.num_simulations, sim.config.num_time_steps))
        rng = np.random.default_rng(0)
        cases_vec, case_beam_df = sim._apply_observation_model(infec, params, rng)
        assert_allclose(cases_vec, 0)

    def test_case_beam_lower_le_upper(self, sim, params):
        """Sorted quantiles must produce non-decreasing beam values."""
        infec = np.full(
            (sim.config.num_simulations, sim.config.num_time_steps), 50.0
        )
        rng = np.random.default_rng(0)
        _, case_beam_df = sim._apply_observation_model(infec, params, rng)

        q_sorted = sorted(sim.config.case_beam_quantiles)
        beam_low = case_beam_df.xs(q_sorted[0], level="quantile").values
        beam_high = case_beam_df.xs(q_sorted[-1], level="quantile").values
        assert np.all(beam_low <= beam_high)

    def test_determinism_with_same_seed(self, sim, params):
        """Same RNG seed → identical cases_vec."""
        infec = np.full(
            (sim.config.num_simulations, sim.config.num_time_steps), 10.0
        )
        c1, _ = sim._apply_observation_model(infec, params, np.random.default_rng(7))
        c2, _ = sim._apply_observation_model(infec, params, np.random.default_rng(7))
        np.testing.assert_array_equal(c1, c2)

    def test_missing_required_params_raises(self, sim, params):
        infec = np.ones((sim.config.num_simulations, sim.config.num_time_steps))
        bad_params = params.drop(columns=["notif_nb_overdispersion"])
        with pytest.raises(ValueError, match="missing"):
            sim._apply_observation_model(infec, bad_params, np.random.default_rng(0))

    def test_negative_population_raises(self, sim, params):
        infec = np.ones((sim.config.num_simulations, sim.config.num_time_steps))
        params["notif_relative_scale"] = 1.0
        population_size = -int(1E5)
        with pytest.raises(ValueError, match="population_size must be non-negative"):
            sim._apply_observation_model(
                infec, params, np.random.default_rng(0),
                population_size=population_size,
            )

    def test_relative_scaling_equivalent_to_absolute_scaling(self, sim, params):
        """Equivalent absolute/relative scaling should produce identical outputs."""
        infec = np.full((sim.config.num_simulations, sim.config.num_time_steps), 10.0)
        pop = int(2e5)
        ref_pop = int(1e5)
        relative_scale = 0.5
        effective_scale = relative_scale * pop / ref_pop

        params_abs = params.copy()
        params_abs["notif_scaling_factor"] = effective_scale

        params_rel = params.copy()
        params_rel["notif_scaling_factor"] = 999.0
        params_rel["notif_relative_scale"] = relative_scale

        cases_abs, beam_abs = sim._apply_observation_model(
            infec, params_abs, np.random.default_rng(123)
        )
        cases_rel, beam_rel = sim._apply_observation_model(
            infec,
            params_rel,
            np.random.default_rng(123),
            population_size=pop,
            reference_population_size=ref_pop,
        )

        np.testing.assert_array_equal(cases_abs, cases_rel)
        pd.testing.assert_frame_equal(beam_abs, beam_rel)

    def test_relative_scaling_overrides_notif_scaling_factor(self, sim, params):
        """When notif_relative_scale is present, notif_scaling_factor is ignored."""
        infec = np.full((sim.config.num_simulations, sim.config.num_time_steps), 10.0)
        pop = int(2e5)
        ref_pop = int(1e5)
        relative_scale = 0.25

        params_with_relative = params.copy()
        params_with_relative["notif_scaling_factor"] = 10.0
        params_with_relative["notif_relative_scale"] = relative_scale

        params_effective = params.copy()
        params_effective["notif_scaling_factor"] = relative_scale * pop / ref_pop

        cases_rel, _ = sim._apply_observation_model(
            infec,
            params_with_relative,
            np.random.default_rng(77),
            population_size=pop,
            reference_population_size=ref_pop,
        )
        cases_effective, _ = sim._apply_observation_model(
            infec,
            params_effective,
            np.random.default_rng(77),
        )

        np.testing.assert_array_equal(cases_rel, cases_effective)

    def test_relative_scaling_requires_population_size(self, sim, params):
        infec = np.ones((sim.config.num_simulations, sim.config.num_time_steps))
        params["notif_relative_scale"] = 1.0

        with pytest.raises(ValueError, match="population_size must be provided"):
            sim._apply_observation_model(
                infec,
                params,
                np.random.default_rng(0),
                population_size=None,
            )

    def test_relative_scaling_with_zero_population_gives_zero_cases(self, sim, params):
        infec = np.full((sim.config.num_simulations, sim.config.num_time_steps), 100.0)
        params["notif_relative_scale"] = 1.0

        cases_vec, _ = sim._apply_observation_model(
            infec,
            params,
            np.random.default_rng(0),
            population_size=0,
            reference_population_size=int(1e5),
        )

        assert_allclose(cases_vec, 0)


# ---------------------------------------------------------------------------
# TestRun — integration
# ---------------------------------------------------------------------------

class TestRun:
    """Integration tests for RenewalSimulator.run()."""

    @pytest.fixture
    def sim(self):
        # gt_max=7, step_dt=7 → gt_max_steps=1 (simplest warm-up)
        return make_simulator(num_sim=3, num_steps=4, step_dt=7, gt_max=7)

    @pytest.fixture
    def params(self, sim):
        return make_params_df(sim.config.num_simulations)

    @pytest.fixture
    def initial_infec(self, sim):
        gt_steps = sim._gt_max_steps  # = 1
        return pd.DataFrame(
            np.ones((sim.config.num_simulations, gt_steps))
        )

    # --- Output type and shapes ---

    def test_returns_simulation_output(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        assert isinstance(out, SimulationOutput)

    def test_infec_df_shape(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        n, s = sim.config.num_simulations, sim.config.num_time_steps
        assert out.infec_df.shape == (n, s)

    def test_cases_df_shape(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        n, s = sim.config.num_simulations, sim.config.num_time_steps
        assert out.cases_df.shape == (n, s)

    def test_case_beam_df_multiindex(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        assert out.case_beam_df.index.names == ["quantile", "i_simulation"]

    # --- Timestamp columns ---

    def test_output_columns_are_timestamps(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        assert isinstance(out.infec_df.columns, pd.DatetimeIndex)
        assert isinstance(out.cases_df.columns, pd.DatetimeIndex)

    def test_first_column_is_sim_start(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        assert out.infec_df.columns[0] == sim.config.temporal.sim_start

    def test_column_spacing_equals_step_dt(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        diffs = out.infec_df.columns[1:] - out.infec_df.columns[:-1]
        expected_delta = pd.Timedelta(days=sim.config.temporal.step_dt)
        assert all(d == expected_delta for d in diffs)

    # --- Config stored on output ---

    def test_output_config_is_same_object(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        assert out.config is sim.config

    # --- Projection mode ---

    def test_projection_mode_scoring_is_none(self, sim, params, initial_infec):
        out = sim.run(params, initial_infec)
        assert out.scoring is None

    # --- Calibration mode ---

    def test_calibration_mode_scoring_not_none(self):
        # sim: 2024-01-01, 2024-01-08, 2024-01-15, 2024-01-22 (step_dt=7, 4 steps)
        # calibration window = full simulation range
        cfg = make_config(
            num_sim=3, num_steps=4, step_dt=7, gt_max=7, mode="calibration",
            sim_start="2024-01-01",
            calibration_start="2024-01-01",
            calibration_end="2024-01-22",
        )
        sim_cal = RenewalSimulator(
            rt_model=LogisticRT(),
            gt_model=ConstantGammaGT(shape=5.0, scale=2.0),
            config=cfg,
        )
        params = make_params_df(sim_cal.config.num_simulations)
        gt_steps = sim_cal._gt_max_steps
        initial = pd.DataFrame(np.ones((sim_cal.config.num_simulations, gt_steps)))
        timestamps = pd.date_range(
            start=cfg.temporal.sim_start,
            periods=cfg.num_time_steps,
            freq=pd.tseries.offsets.Day(cfg.temporal.step_dt),
        )
        obs = pd.Series(np.arange(1, cfg.num_time_steps + 1, dtype=float), index=timestamps)
        out = sim_cal.run(params, initial, observations_sr=obs)
        assert out.scoring is not None

    def test_calibration_mode_wis_shape(self):
        # sim: 6 weekly steps from 2024-01-01
        # calibration window: steps 1-3 (2024-01-08 to 2024-01-22) → 3 cal steps
        num_sim, num_steps = 5, 6
        cfg = make_config(
            num_sim=num_sim, num_steps=num_steps, step_dt=7, gt_max=7, mode="calibration",
            sim_start="2024-01-01",
            calibration_start="2024-01-08",
            calibration_end="2024-01-22",
        )
        sim_cal = RenewalSimulator(
            rt_model=LogisticRT(),
            gt_model=ConstantGammaGT(shape=5.0, scale=2.0),
            config=cfg,
        )
        params = make_params_df(sim_cal.config.num_simulations)
        gt_steps = sim_cal._gt_max_steps
        initial = pd.DataFrame(np.ones((num_sim, gt_steps)))
        # Observations covering all 6 simulation steps (wider than calibration window)
        all_timestamps = pd.date_range(
            start=cfg.temporal.sim_start,
            periods=num_steps,
            freq=pd.tseries.offsets.Day(cfg.temporal.step_dt),
        )
        obs = pd.Series(np.ones(num_steps, dtype=float), index=all_timestamps)
        out = sim_cal.run(params, initial, observations_sr=obs)
        # Only the 3 timestamps inside [calibration_start, calibration_end] are scored
        assert out.scoring.wis_array.shape == (num_sim, 3)

    # --- Validation errors ---

    def test_validation_wrong_params_rows(self, sim, initial_infec):
        wrong_params = make_params_df(sim.config.num_simulations + 5)
        with pytest.raises(ValueError, match="params_df"):
            sim.run(wrong_params, initial_infec)

    def test_validation_wrong_initial_infec_shape(self, sim, params):
        bad_initial = pd.DataFrame(
            np.ones((sim.config.num_simulations, sim._gt_max_steps + 1))
        )
        with pytest.raises(ValueError, match="initial_infec_df"):
            sim.run(params, bad_initial)

    def test_validation_calibration_requires_observations(self):
        cfg = make_config(
            mode="calibration",
            calibration_start="2024-01-01",
            calibration_end="2024-01-22",
        )
        sim_cal = RenewalSimulator(
            rt_model=LogisticRT(),
            gt_model=ConstantGammaGT(shape=5.0, scale=2.0),
            config=cfg,
        )
        params = make_params_df(sim_cal.config.num_simulations)
        initial = pd.DataFrame(np.ones((sim_cal.config.num_simulations, sim_cal._gt_max_steps)))
        with pytest.raises(ValueError, match="observations_sr"):
            sim_cal.run(params, initial)

    def test_validation_calibration_bounds_required(self):
        """mode='calibration' without calibration_start/end raises ValueError."""
        cfg = make_config(mode="calibration")  # no calibration_start/calibration_end
        sim_cal = RenewalSimulator(
            rt_model=LogisticRT(),
            gt_model=ConstantGammaGT(shape=5.0, scale=2.0),
            config=cfg,
        )
        params = make_params_df(cfg.num_simulations)
        initial = pd.DataFrame(np.ones((cfg.num_simulations, sim_cal._gt_max_steps)))
        timestamps = pd.date_range(
            start=cfg.temporal.sim_start,
            periods=cfg.num_time_steps,
            freq=pd.tseries.offsets.Day(cfg.temporal.step_dt),
        )
        obs = pd.Series(np.ones(cfg.num_time_steps, dtype=float), index=timestamps)
        with pytest.raises(ValueError, match="calibration_start"):
            sim_cal.run(params, initial, observations_sr=obs)

    def test_validation_calibration_empty_window(self):
        """No observations inside the calibration window raises ValueError."""
        # sim steps are in 2024; calibration window is far in the future
        cfg = make_config(
            mode="calibration",
            calibration_start="2030-01-01",
            calibration_end="2030-12-31",
        )
        sim_cal = RenewalSimulator(
            rt_model=LogisticRT(),
            gt_model=ConstantGammaGT(shape=5.0, scale=2.0),
            config=cfg,
        )
        params = make_params_df(cfg.num_simulations)
        initial = pd.DataFrame(np.ones((cfg.num_simulations, sim_cal._gt_max_steps)))
        timestamps = pd.date_range(
            start=cfg.temporal.sim_start,
            periods=cfg.num_time_steps,
            freq=pd.tseries.offsets.Day(cfg.temporal.step_dt),
        )
        obs = pd.Series(np.ones(cfg.num_time_steps, dtype=float), index=timestamps)
        with pytest.raises(ValueError, match="No observation data"):
            sim_cal.run(params, initial, observations_sr=obs)

    def test_validation_calibration_misaligned_timestamps(self):
        """Calibration obs timestamps not present in simulation raise ValueError."""
        cfg = make_config(
            mode="calibration",
            sim_start="2024-01-01",
            calibration_start="2024-01-01",
            calibration_end="2024-01-22",
        )
        sim_cal = RenewalSimulator(
            rt_model=LogisticRT(),
            gt_model=ConstantGammaGT(shape=5.0, scale=2.0),
            config=cfg,
        )
        params = make_params_df(cfg.num_simulations)
        initial = pd.DataFrame(np.ones((cfg.num_simulations, sim_cal._gt_max_steps)))
        # Observations shifted by 1 day — not aligned with weekly simulation steps
        sim_timestamps = pd.date_range(
            start=cfg.temporal.sim_start,
            periods=cfg.num_time_steps,
            freq=pd.tseries.offsets.Day(cfg.temporal.step_dt),
        )
        obs_timestamps = sim_timestamps + pd.Timedelta(days=1)
        obs = pd.Series(np.ones(cfg.num_time_steps, dtype=float), index=obs_timestamps)
        with pytest.raises(ValueError, match="absent from the simulation"):
            sim_cal.run(params, initial, observations_sr=obs)

    # --- t_start / zero_date independence ---

    def test_zero_date_offset_does_not_change_infection_output(self):
        """Shifting zero_date (with same sim_start) should not change infections.

        The RT window shifts accordingly via t_start, so the effective R(t)
        values at the warm-up and simulation steps remain the same.
        """
        # Build two configs with different zero_dates but same sim_start
        base_date = pd.Timestamp("2024-06-01")
        cfg_a = make_config(
            num_sim=3, num_steps=3, step_dt=7, gt_max=7,
            zero_date="2024-01-01", sim_start="2024-06-01",
        )
        cfg_b = make_config(
            num_sim=3, num_steps=3, step_dt=7, gt_max=7,
            zero_date="2024-06-01",   # zero_date == sim_start (like the prototype)
            sim_start="2024-06-01",
        )

        # R(t) parameters expressed relative to zero_date_b (= sim_start)
        # Use the same absolute day values for R(t) in both configs.
        # For cfg_a: zero_date is 2024-01-01 and sim_start is 2024-06-01
        #   sim_start_day = (2024-06-01 - 2024-01-01).days = 152
        #   t_start for RT = 152 - 1*7 = 145  (gt_max_steps=1)
        # For cfg_b: zero_date == sim_start → sim_start_day = 0
        #   t_start for RT = 0 - 1*7 = -7
        #
        # We want R(t) to look the same at the actual calendar steps.
        # The easiest check: use R(t)=constant → output is independent of the
        # time axis origin.  We compare infec_df values.

        rng_seed = 0
        gt_model = ConstantGammaGT(shape=5.0, scale=2.0)
        rt_model = LogisticRT()

        sim_a = RenewalSimulator(rt_model=rt_model, gt_model=gt_model, config=cfg_a)
        sim_b = RenewalSimulator(rt_model=rt_model, gt_model=gt_model, config=cfg_b)

        # Force R(t) = constant 1.2 by using a very wide window
        params_a = make_params_df(
            3, roff=1.2, start=-1e6, end=1e6, center=0.0, width=1e5, rmin=1.2, rmax=1.2
        )
        params_b = make_params_df(
            3, roff=1.2, start=-1e6, end=1e6, center=0.0, width=1e5, rmin=1.2, rmax=1.2
        )

        initial = pd.DataFrame(np.ones((3, 1)))
        out_a = sim_a.run(params_a, initial)
        out_b = sim_b.run(params_b, initial)

        # The infection arrays should be identical (same effective R(t))
        assert_allclose(out_a.infec_df.values, out_b.infec_df.values, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestFactory
# ---------------------------------------------------------------------------

class TestFactory:
    """Tests for RenewalSimulator.from_config_dict()."""

    def test_from_config_dict_builds_nested_dataclasses(self):
        config_dict = {
            "reproduction_number": {
                "model": "logistic",
                "params": {
                    "rt_logist_roff": 1.0,
                },
            },
            "generation_time": {
                "model": "constant_gamma",
                "params": {
                    "shape": 6.0,
                    "scale": 1.5,
                },
            },
            "simulation": {
                "mode": "calibration",
                "num_simulations": 7,
                "num_time_steps": 12,
                "gt_max": 35,
                "rng_seed": 123,
            },
            "temporal": {
                "zero_date": "2024-01-01",
                "sim_start": 14.0,          # float days since zero_date
                "calibration_start": 7.0,
                "calibration_end": "2024-02-05",
                "step_dt": 7,
            },
            "location": {
                "location_id_variable": "uf",
                "location_id": "DF",
            },
            "observation_model": {
                "params": {
                    "notif_nb_overdispersion": 6.5,
                    "notif_scaling_factor": 2.0,
                }
            },
            "sampling": {
                "method": "lhs",
                "param_ranges": {
                    "rt_logist_width": [1.0, 80.0],
                },
            },
            "scoring": {
                "case_beam_quantiles": [0.1, 0.5, 0.9],
            },
        }

        sim = RenewalSimulator.from_config_dict(config_dict=config_dict)

        cfg = sim.config
        assert isinstance(sim.rt_model, LogisticRT)
        assert isinstance(sim.gt_model, ConstantGammaGT)
        assert sim.gt_model.shape == pytest.approx(6.0)
        assert sim.gt_model.scale == pytest.approx(1.5)
        assert cfg.mode == "calibration"
        assert cfg.num_simulations == 7
        assert cfg.num_time_steps == 12
        assert cfg.gt_max == 35
        assert cfg.rng_seed == 123
        assert cfg.temporal.zero_date == pd.Timestamp("2024-01-01")
        assert cfg.temporal.sim_start == pd.Timestamp("2024-01-15")
        assert cfg.temporal.calibration_start == pd.Timestamp("2024-01-08")
        assert cfg.temporal.calibration_end == pd.Timestamp("2024-02-05")
        assert cfg.temporal.step_dt == 7
        assert cfg.location.location_id_variable == "uf"
        assert cfg.location.location_id == "DF"
        assert cfg.notif_nb_overdispersion == pytest.approx(6.5)
        assert cfg.notif_scaling_factor == pytest.approx(2.0)
        assert cfg.case_beam_quantiles == [0.1, 0.5, 0.9]
        assert cfg.sampling is not None
        assert cfg.sampling.method == "lhs"
        assert cfg.sampling.param_ranges["rt_logist_width"] == [1.0, 80.0]
        assert cfg.sampling.rt_params["rt_logist_roff"] == pytest.approx(1.0)
        assert cfg.sampling.observation_params["notif_nb_overdispersion"] == pytest.approx(6.5)
        assert cfg.initial_infections.method == "ones"

    def test_from_config_dict_uses_defaults_when_sections_missing(self):
        sim = RenewalSimulator.from_config_dict(
            config_dict={"simulation": {"num_simulations": 1000}}
        )
        defaults = SimulationConfig()
        cfg = sim.config

        assert isinstance(sim.rt_model, LogisticRT)
        assert isinstance(sim.gt_model, ConstantGammaGT)
        assert sim.gt_model.shape == pytest.approx(10.0)
        assert sim.gt_model.scale == pytest.approx(1.8)
        assert cfg.mode == defaults.mode
        assert cfg.num_simulations == defaults.num_simulations
        assert cfg.num_time_steps == defaults.num_time_steps
        assert cfg.gt_max == defaults.gt_max
        assert cfg.temporal.zero_date == defaults.temporal.zero_date
        assert cfg.temporal.sim_start == defaults.temporal.sim_start
        assert cfg.temporal.step_dt == defaults.temporal.step_dt
        assert cfg.location.location_id_variable == defaults.location.location_id_variable
        assert cfg.location.location_id == defaults.location.location_id
        assert cfg.notif_nb_overdispersion == defaults.notif_nb_overdispersion
        assert cfg.notif_scaling_factor == defaults.notif_scaling_factor
        assert cfg.case_beam_quantiles == defaults.case_beam_quantiles
        assert cfg.sampling is not None
        assert cfg.sampling.method == "lhs"
        assert cfg.sampling.param_ranges == {}
        assert cfg.initial_infections.method == "ones"
        assert cfg.rng_seed == defaults.rng_seed

    def test_from_config_dict_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="simulation.mode"):
            RenewalSimulator.from_config_dict(
                config_dict={"simulation": {"mode": "invalid", "num_simulations": 10}},
            )

    def test_from_config_dict_invalid_rt_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported reproduction_number.model"):
            RenewalSimulator.from_config_dict(
                config_dict={
                    "simulation": {"num_simulations": 10},
                    "reproduction_number": {"model": "unknown_rt"},
                },
            )

    def test_from_config_dict_invalid_gt_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported generation_time.model"):
            RenewalSimulator.from_config_dict(
                config_dict={
                    "simulation": {"num_simulations": 10},
                    "generation_time": {"model": "unknown_gt"},
                },
            )
