"""Unit tests for outbreak_dynamics.scoring.

Synthetic data strategy
-----------------------
- ``nbinom_ppf_cf``:  Compare against ``scipy.stats.nbinom.ppf`` on a grid of
  (q, n, p) values.  Allow a generous tolerance because the CF expansion is an
  approximation; verify monotonicity and floor at 0.

- ``wis_score_vectorized``:  Construct minimal case beams (3 quantiles: 0.025,
  0.5, 0.975) from known constants.  Check perfect-forecast → small WIS,
  wide-interval → large WIS, shape, and non-negativity.

- ``rmse_vectorized`` / ``smape_vectorized``:  Use 2-simulation, 4-step
  DataFrames with known values; check exact outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from inframind_proteus.outbreak_dynamics.scoring import (
    nb_loglikelihood_vectorized,
    nbinom_ppf_cf,
    rmse_vectorized,
    smape_vectorized,
    wis_score_vectorized,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case_beam(
    medians: np.ndarray,
    half_widths: np.ndarray,
    n_sims: int,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Build a synthetic MultiIndex case-beam DataFrame.

    Each simulation shares the same trajectory (medians ± half_widths).

    Parameters
    ----------
    medians:
        Shape ``(num_time_steps,)`` median prediction.
    half_widths:
        Shape ``(num_time_steps,)`` half-width of the 95 % PI.
    n_sims:
        Number of simulations (rows per quantile).
    quantiles:
        Quantile levels to include.  Defaults to [0.025, 0.5, 0.975].
    """
    if quantiles is None:
        quantiles = [0.025, 0.5, 0.975]

    time_steps = pd.RangeIndex(len(medians))
    rows = {}
    for q in quantiles:
        if q < 0.5:
            vals = medians - half_widths
        elif q > 0.5:
            vals = medians + half_widths
        else:
            vals = medians.copy()
        for sim_i in range(n_sims):
            rows[(q, sim_i)] = vals

    idx = pd.MultiIndex.from_tuples(rows.keys(), names=["quantile", "i_simulation"])
    return pd.DataFrame(list(rows.values()), index=idx, columns=time_steps)


# ---------------------------------------------------------------------------
# nbinom_ppf_cf
# ---------------------------------------------------------------------------

class TestNbinomPpfCf:

    def test_output_shape_scalar_q(self):
        n = np.array([5.0, 10.0, 20.0])
        p = np.array([0.5, 0.3, 0.7])
        result = nbinom_ppf_cf(0.5, n, p)
        assert result.shape == (3,)

    def test_output_shape_array_q(self):
        # When q is also an array, broadcasting should work
        n = np.full(4, 10.0)
        p = np.full(4, 0.5)
        result = nbinom_ppf_cf(np.array([0.1, 0.5, 0.9, 0.99]), n, p)
        assert result.shape == (4,)

    def test_non_negative(self):
        n = np.array([0.5, 1.0, 2.0])
        p = np.array([0.9, 0.5, 0.1])
        result = nbinom_ppf_cf(0.01, n, p)
        assert np.all(result >= 0.0)

    def test_monotone_in_q(self):
        """Higher quantile → higher value."""
        n = np.full(3, 10.0)
        p = np.full(3, 0.4)
        qs = [0.1, 0.5, 0.9]
        vals = [nbinom_ppf_cf(q, n, p) for q in qs]
        for lo, hi in zip(vals, vals[1:]):
            assert np.all(hi >= lo)

    def test_median_close_to_scipy(self):
        """CF median should be within 2 counts of scipy for moderate n."""
        n_vals = np.array([5.0, 10.0, 30.0, 50.0])
        p_vals = np.array([0.3, 0.5, 0.4, 0.6])
        exact = scipy.stats.nbinom.ppf(0.5, n_vals, p_vals).astype(float)
        approx = nbinom_ppf_cf(0.5, n_vals, p_vals, continuity=False)
        assert np.all(np.abs(approx - exact) <= 2.0)

    def test_tail_quantile_exceeds_median(self):
        n = np.array([10.0])
        p = np.array([0.5])
        med = nbinom_ppf_cf(0.5, n, p)
        q95 = nbinom_ppf_cf(0.95, n, p)
        assert q95 > med

    def test_continuity_correction_adds_half(self):
        """continuity=True result should be continuity=False result + 0.5."""
        n = np.array([20.0, 20.0])
        p = np.array([0.4, 0.6])
        without = nbinom_ppf_cf(0.5, n, p, continuity=False)
        with_ = nbinom_ppf_cf(0.5, n, p, continuity=True)
        np.testing.assert_allclose(with_, without + 0.5)


# ---------------------------------------------------------------------------
# wis_score_vectorized
# ---------------------------------------------------------------------------

class TestWisScoreVectorized:

    def test_output_shape(self):
        medians = np.array([10.0, 20.0, 15.0, 5.0])
        hw = np.array([3.0, 5.0, 4.0, 2.0])
        beam = _make_case_beam(medians, hw, n_sims=8)
        obs = pd.Series(medians, index=beam.columns)
        wis = wis_score_vectorized(beam, obs)
        assert wis.shape == (8, 4)

    def test_non_negative(self):
        medians = np.array([10.0, 20.0, 15.0])
        hw = np.ones(3) * 5.0
        beam = _make_case_beam(medians, hw, n_sims=5)
        obs = pd.Series(medians + 2.0, index=beam.columns)
        wis = wis_score_vectorized(beam, obs)
        assert np.all(wis >= 0.0)

    def test_perfect_forecast_small_wis(self):
        """When observed == median and obs is within PI → WIS driven only by
        interval width (sharpness term), which cannot be zero unless PI=0.
        But WIS is still low compared to a wide-interval baseline."""
        medians = np.array([10.0, 10.0, 10.0])
        hw_narrow = np.ones(3) * 1.0
        hw_wide = np.ones(3) * 100.0
        n_sims = 4
        obs = pd.Series(medians, index=pd.RangeIndex(3))

        beam_narrow = _make_case_beam(medians, hw_narrow, n_sims)
        beam_wide = _make_case_beam(medians, hw_wide, n_sims)

        wis_narrow = wis_score_vectorized(beam_narrow, obs)
        wis_wide = wis_score_vectorized(beam_wide, obs)

        assert np.mean(wis_narrow) < np.mean(wis_wide)

    def test_obs_outside_pi_raises_wis(self):
        """When obs is far outside PI, WIS must increase."""
        medians = np.array([10.0, 10.0])
        hw = np.ones(2) * 1.0  # tight PI: [9, 11]
        beam = _make_case_beam(medians, hw, n_sims=3)

        obs_inside = pd.Series(medians, index=beam.columns)
        obs_outside = pd.Series([100.0, 100.0], index=beam.columns)

        wis_in = wis_score_vectorized(beam, obs_inside)
        wis_out = wis_score_vectorized(beam, obs_outside)

        assert np.mean(wis_out) > np.mean(wis_in)

    def test_custom_alphas_and_weights(self):
        medians = np.array([5.0, 5.0, 5.0])
        hw = np.ones(3) * 2.0
        beam = _make_case_beam(medians, hw, n_sims=2)
        obs = pd.Series(medians, index=beam.columns)

        alphas = np.array([0.05])  # Only 95 % PI
        weights = np.array([1.0])
        wis = wis_score_vectorized(beam, obs, alphas=alphas, weights=weights)
        assert wis.shape == (2, 3)

    def test_raises_on_missing_median(self):
        """If 0.5 quantile is absent, must raise AssertionError."""
        medians = np.array([10.0, 20.0])
        hw = np.ones(2) * 3.0
        # Build beam without 0.5
        beam = _make_case_beam(medians, hw, n_sims=2, quantiles=[0.025, 0.975])
        obs = pd.Series(medians, index=beam.columns)
        with pytest.raises(AssertionError):
            wis_score_vectorized(beam, obs)

    def test_raises_on_column_mismatch(self):
        medians = np.array([10.0, 20.0])
        hw = np.ones(2) * 3.0
        beam = _make_case_beam(medians, hw, n_sims=2)
        bad_obs = pd.Series([10.0, 20.0, 30.0])  # Wrong number of steps
        with pytest.raises(AssertionError):
            wis_score_vectorized(beam, bad_obs)

    def test_identical_simulations_same_wis(self):
        """All simulations are identical → all WIS rows must be equal."""
        medians = np.array([8.0, 12.0, 10.0])
        hw = np.ones(3) * 2.0
        beam = _make_case_beam(medians, hw, n_sims=6)
        obs = pd.Series([7.0, 13.0, 9.0], index=beam.columns)
        wis = wis_score_vectorized(beam, obs)
        for row in wis:
            np.testing.assert_allclose(row, wis[0])


# ---------------------------------------------------------------------------
# rmse_vectorized
# ---------------------------------------------------------------------------

class TestRmseVectorized:

    def _make_sim_df(self, data: np.ndarray) -> pd.DataFrame:
        """data shape: (n_sims, n_steps)."""
        return pd.DataFrame(data, columns=pd.RangeIndex(data.shape[1]))

    def test_output_shape(self):
        sim = self._make_sim_df(np.ones((5, 4)))
        obs = pd.Series(np.ones(4))
        result = rmse_vectorized(sim, obs)
        assert result.shape == (5,)

    def test_perfect_forecast_zero_rmse(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sim = self._make_sim_df(data)
        obs = pd.Series(data[0], index=sim.columns)  # Matches first row
        result = rmse_vectorized(sim, obs)
        assert result[0] == pytest.approx(0.0)

    def test_known_value(self):
        """RMSE for [0, 0] vs [3, 4] = sqrt((9+16)/2) = sqrt(12.5)."""
        data = np.array([[3.0, 4.0]])
        sim = self._make_sim_df(data)
        obs = pd.Series([0.0, 0.0], index=sim.columns)
        result = rmse_vectorized(sim, obs)
        assert result[0] == pytest.approx(np.sqrt(12.5))

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 100, size=(10, 8)).astype(float)
        sim = self._make_sim_df(data)
        obs = pd.Series(rng.integers(0, 100, size=8).astype(float), index=sim.columns)
        result = rmse_vectorized(sim, obs)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# smape_vectorized
# ---------------------------------------------------------------------------

class TestSmapeVectorized:

    def _make_sim_df(self, data: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(data, columns=pd.RangeIndex(data.shape[1]))

    def test_output_shape(self):
        sim = self._make_sim_df(np.ones((5, 4)))
        obs = pd.Series(np.ones(4))
        result = smape_vectorized(sim, obs)
        assert result.shape == (5,)

    def test_perfect_forecast_zero_smape(self):
        data = np.array([[10.0, 20.0, 30.0]])
        sim = self._make_sim_df(data)
        obs = pd.Series(data[0], index=sim.columns)
        result = smape_vectorized(sim, obs)
        assert result[0] == pytest.approx(0.0)

    def test_known_value(self):
        """sMAPE for sim=[0], obs=[10]: 2*|0-10|/(|0|+|10|) = 2."""
        data = np.array([[0.0]])
        sim = self._make_sim_df(data)
        obs = pd.Series([10.0], index=sim.columns)
        result = smape_vectorized(sim, obs)
        assert result[0] == pytest.approx(2.0)

    def test_both_zero_treated_as_zero_error(self):
        """When both sim and obs are 0, the term should be 0 (not NaN)."""
        data = np.array([[0.0, 5.0]])
        sim = self._make_sim_df(data)
        obs = pd.Series([0.0, 5.0], index=sim.columns)
        result = smape_vectorized(sim, obs)
        assert result[0] == pytest.approx(0.0)
        assert np.isfinite(result[0])

    def test_bounded_0_2(self):
        """sMAPE is always in [0, 2]."""
        rng = np.random.default_rng(7)
        data = rng.integers(0, 200, size=(20, 10)).astype(float)
        sim = self._make_sim_df(data)
        obs = pd.Series(rng.integers(0, 200, size=10).astype(float), index=sim.columns)
        result = smape_vectorized(sim, obs)
        assert np.all(result >= 0.0)
        assert np.all(result <= 2.0)


# ---------------------------------------------------------------------------
# nb_loglikelihood_vectorized
# ---------------------------------------------------------------------------

class TestNbLoglikelihoodVectorized:
    def test_loglikelihood_matches_scipy_loop(self):
        """Check vectorized NB log-likelihood against a direct SciPy loop."""
        # --- Setup: 2 simulations, 3 time steps ---
        simulations_df = pd.DataFrame(
            data=[
                [10.0, 20.0, 30.0],  # Predictions for sim 0
                [12.0, 22.0, 32.0],  # Predictions for sim 1
            ],
            columns=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        )
        simulations_df.index.name = "i_simulation"
        simulations_df.columns.name = "t"

        observations_sr = pd.Series(
            data=[15, 25, 35],
            index=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        )
        observations_sr.index.name = "t"

        overdisp = np.array([100.0, 110.0])  # n parameter for each sim

        # --- Action ---
        result = nb_loglikelihood_vectorized(
            simulations_df=simulations_df,
            observations_sr=observations_sr,
            overdisp=overdisp,
        )

        # --- Verification ---
        # Manually calculate expected values using a loop over simulations.
        expected_ll = []
        for i_sim in range(2):
            pred = simulations_df.iloc[i_sim].values
            obs = observations_sr.values
            n = overdisp[i_sim]
            p = n / (pred + n)
            ll_components = scipy.stats.nbinom.logpmf(k=obs, n=n, p=p)
            expected_ll.append(np.sum(ll_components))

        expected = np.array(expected_ll)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
