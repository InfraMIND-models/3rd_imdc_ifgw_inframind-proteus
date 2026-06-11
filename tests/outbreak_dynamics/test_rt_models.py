"""Unit tests for outbreak_dynamics.rt_models.

Design notes
------------
- LogisticRT.generate returns shape (num_simulations, num_time_steps).
- Outside [rt_logist_start, rt_logist_end) → R(t) = rt_logist_roff.
- Inside the window → declining logistic from rmax toward rmin.
- At t = rt_logist_center the logistic equals exactly (rmax + rmin) / 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from inframind_proteus.outbreak_dynamics.rt_models import BaseRT, LogisticRT, EnvelopedLogisticRT


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_params(
    roff=1.0,
    start=14.0,
    end=140.0,
    center=70.0,
    width=14.0,
    rmin=0.5,
    rmax=2.5,
    n=3,
) -> pd.DataFrame:
    """Return a params_df with ``n`` identical rows (easy to vary one at a time)."""
    return pd.DataFrame(
        {
            "rt_logist_roff":   [roff]   * n,
            "rt_logist_start":  [start]  * n,
            "rt_logist_end":    [end]    * n,
            "rt_logist_center": [center] * n,
            "rt_logist_width":  [width]  * n,
            "rt_logist_rmin":   [rmin]   * n,
            "rt_logist_rmax":   [rmax]   * n,
        }
    )


# ---------------------------------------------------------------------------
# BaseRT interface
# ---------------------------------------------------------------------------

class TestBaseRTInterface:

    def test_logistic_is_subclass(self):
        assert issubclass(LogisticRT, BaseRT)

    def test_required_params_contains_end(self):
        assert "rt_logist_end" in LogisticRT.required_params

    def test_required_params_complete(self):
        expected = {
            "rt_logist_roff", "rt_logist_start", "rt_logist_end",
            "rt_logist_center", "rt_logist_width", "rt_logist_rmin",
            "rt_logist_rmax",
        }
        assert set(LogisticRT.required_params) == expected


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------

class TestValidateParams:

    def test_raises_on_missing_column(self):
        df = make_params().drop(columns=["rt_logist_end"])
        with pytest.raises(ValueError, match="rt_logist_end"):
            LogisticRT().validate_params(df)

    def test_raises_lists_all_missing(self):
        df = make_params().drop(columns=["rt_logist_end", "rt_logist_rmax"])
        with pytest.raises(ValueError):
            LogisticRT().validate_params(df)

    def test_passes_with_all_columns(self):
        LogisticRT().validate_params(make_params())  # should not raise


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_shape_single_simulation(self):
        rt = LogisticRT().generate(make_params(n=1), num_time_steps=20, step_dt=7)
        assert rt.shape == (1, 20)

    def test_shape_many_simulations(self):
        rt = LogisticRT().generate(make_params(n=50), num_time_steps=40, step_dt=7)
        assert rt.shape == (50, 40)

    def test_dtype_float(self):
        rt = LogisticRT().generate(make_params(n=2), num_time_steps=10, step_dt=7)
        assert np.issubdtype(rt.dtype, np.floating)


# ---------------------------------------------------------------------------
# Off-season / boundary behaviour
# ---------------------------------------------------------------------------

class TestOffSeasonBehaviour:

    def test_before_start_equals_roff(self):
        """All steps strictly before rt_logist_start must equal roff."""
        step_dt = 7
        # start=70 → first active step is index 10 (70/7=10)
        params = make_params(roff=1.2, start=70.0, end=280.0, n=4)
        rt = LogisticRT().generate(params, num_time_steps=50, step_dt=step_dt)
        # Steps 0..9 correspond to days 0..63 < 70
        assert_allclose(rt[:, :10], 1.2)

    def test_after_end_equals_roff(self):
        """All steps at or beyond rt_logist_end must equal roff."""
        step_dt = 7
        # end=140 → step index 20 is day 140, which is >= end → roff
        params = make_params(roff=0.8, start=0.0, end=140.0, n=4)
        rt = LogisticRT().generate(params, num_time_steps=30, step_dt=step_dt)
        assert_allclose(rt[:, 20:], 0.8)

    def test_zero_width_window_all_roff(self):
        """When start == end the active window is empty → all roff."""
        params = make_params(roff=1.5, start=50.0, end=50.0, n=3)
        rt = LogisticRT().generate(params, num_time_steps=20, step_dt=7)
        assert_allclose(rt, 1.5)


# ---------------------------------------------------------------------------
# Logistic shape inside the active window
# ---------------------------------------------------------------------------

class TestLogisticShape:

    def test_at_center_equals_midpoint(self):
        """At t = center, logistic = (rmax + rmin) / 2."""
        step_dt = 7
        center = 70.0  # step index 10
        rmin, rmax = 0.4, 2.4
        params = make_params(
            start=0.0, end=280.0, center=center,
            rmin=rmin, rmax=rmax, n=2,
        )
        rt = LogisticRT().generate(params, num_time_steps=20, step_dt=step_dt)
        expected_midpoint = (rmax + rmin) / 2.0
        assert_allclose(rt[:, 10], expected_midpoint, rtol=1e-6)

    def test_monotonically_decreasing_inside_window(self):
        """Within the active window R(t) should be non-increasing."""
        step_dt = 7
        params = make_params(
            start=0.0, end=700.0, center=140.0,
            width=28.0, rmin=0.3, rmax=3.0, n=5,
        )
        rt = LogisticRT().generate(params, num_time_steps=100, step_dt=step_dt)
        # Check the active window columns (days 0..693 < 700 → steps 0..99)
        diffs = np.diff(rt, axis=1)
        assert np.all(diffs <= 1e-10), "R(t) must be non-increasing inside the window"

    def test_near_rmax_early_in_window(self):
        """Far before the center, R(t) should be close to rmax."""
        step_dt = 7
        # center=280, width=7 → at t=0 the logistic ≈ rmax
        params = make_params(
            start=0.0, end=700.0, center=280.0,
            width=7.0, rmin=0.2, rmax=3.0, n=2,
        )
        rt = LogisticRT().generate(params, num_time_steps=5, step_dt=step_dt)
        assert_allclose(rt[:, 0], 3.0, atol=1e-3)

    def test_near_rmin_late_in_window(self):
        """Far after the center, R(t) should be close to rmin."""
        step_dt = 7
        # center=7, width=7 → at t=200 the logistic ≈ rmin
        params = make_params(
            start=0.0, end=300.0, center=7.0,
            width=7.0, rmin=0.5, rmax=3.0, n=2,
        )
        rt = LogisticRT().generate(params, num_time_steps=43, step_dt=step_dt)
        # step 42 → day 294, far past center 7 with width 7 → nearly rmin
        assert_allclose(rt[:, 42], 0.5, atol=1e-3)


# ---------------------------------------------------------------------------
# Per-simulation heterogeneity
# ---------------------------------------------------------------------------

class TestPerSimulationVariation:

    def test_different_roff_per_row(self):
        """Each simulation should use its own roff outside the window."""
        params = pd.DataFrame(
            {
                "rt_logist_roff":   [1.0, 2.0, 3.0],
                "rt_logist_start":  [100.0, 100.0, 100.0],
                "rt_logist_end":    [200.0, 200.0, 200.0],
                "rt_logist_center": [150.0, 150.0, 150.0],
                "rt_logist_width":  [14.0, 14.0, 14.0],
                "rt_logist_rmin":   [0.5, 0.5, 0.5],
                "rt_logist_rmax":   [2.5, 2.5, 2.5],
            }
        )
        rt = LogisticRT().generate(params, num_time_steps=10, step_dt=7)
        # All 10 steps are before start (day 63 < 100)
        assert_allclose(rt[0, :], 1.0)
        assert_allclose(rt[1, :], 2.0)
        assert_allclose(rt[2, :], 3.0)

    def test_different_end_per_row(self):
        """Simulations with different rt_logist_end should return roff at different points."""
        step_dt = 7
        params = pd.DataFrame(
            {
                "rt_logist_roff":   [1.0, 1.0],
                "rt_logist_start":  [0.0, 0.0],
                "rt_logist_end":    [49.0, 98.0],   # 7 steps vs 14 steps
                "rt_logist_center": [200.0, 200.0],  # far away → near rmax throughout
                "rt_logist_width":  [14.0, 14.0],
                "rt_logist_rmin":   [0.3, 0.3],
                "rt_logist_rmax":   [2.5, 2.5],
            }
        )
        rt = LogisticRT().generate(params, num_time_steps=20, step_dt=step_dt)
        # Sim 0: step 7 is day 49 → roff (>= end=49)
        assert_allclose(rt[0, 7], 1.0)
        # Sim 1: step 7 is day 49 < 98 → still in window (not roff)
        assert rt[1, 7] != 1.0


# ---------------------------------------------------------------------------
# EnvelopedLogisticRT Tests
# ---------------------------------------------------------------------------

def make_envelope_params(
    start=14.0,
    dt_center=56.0,
    dt_end=126.0,
    w_center=14.0,
    w_env=7.0,
    r_low=0.5,
    r_high=2.5,
    n=3,
) -> pd.DataFrame:
    """Return a params_df for EnvelopedLogisticRT with ``n`` identical rows."""
    return pd.DataFrame(
        {
            "rt_logist_start":     [start]     * n,
            "rt_logist_dt_center": [dt_center] * n,
            "rt_logist_dt_end":    [dt_end]    * n,
            "rt_logist_w_center":  [w_center]  * n,
            "rt_logist_w_env":     [w_env]     * n,
            "rt_logist_r_low":     [r_low]     * n,
            "rt_logist_r_high":    [r_high]    * n,
        }
    )


class TestEnvelopedLogisticRTInterface:

    def test_enveloped_is_subclass(self):
        assert issubclass(EnvelopedLogisticRT, BaseRT)

    def test_required_params_complete(self):
        expected = {
            "rt_logist_start", "rt_logist_dt_center", "rt_logist_dt_end",
            "rt_logist_w_center", "rt_logist_w_env", "rt_logist_r_low",
            "rt_logist_r_high",
        }
        assert set(EnvelopedLogisticRT.required_params) == expected


class TestEnvelopedValidateParams:

    def test_raises_on_missing_column(self):
        df = make_envelope_params().drop(columns=["rt_logist_dt_center"])
        with pytest.raises(ValueError, match="rt_logist_dt_center"):
            EnvelopedLogisticRT().validate_params(df)

    def test_passes_with_all_columns(self):
        EnvelopedLogisticRT().validate_params(make_envelope_params())  # should not raise


class TestEnvelopedOutputShape:

    def test_shape_single_simulation(self):
        rt = EnvelopedLogisticRT().generate(
            make_envelope_params(n=1), num_time_steps=20, step_dt=7
        )
        assert rt.shape == (1, 20)

    def test_shape_many_simulations(self):
        rt = EnvelopedLogisticRT().generate(
            make_envelope_params(n=50), num_time_steps=40, step_dt=7
        )
        assert rt.shape == (50, 40)

    def test_dtype_float(self):
        rt = EnvelopedLogisticRT().generate(
            make_envelope_params(n=2), num_time_steps=10, step_dt=7
        )
        assert np.issubdtype(rt.dtype, np.floating)


class TestEnvelopedOffSeasonBehaviour:

    def test_baseline_near_one_before_season(self):
        """Far before the active season starts, R(t) should be near 1.0."""
        step_dt = 7
        # start=140 → first 20 steps (days 0..133) are before season
        params = make_envelope_params(start=140.0, dt_end=126.0, w_env=7.0, n=3)
        rt = EnvelopedLogisticRT().generate(params, num_time_steps=25, step_dt=step_dt)
        # Before season, envelope E(t) ≈ 0, so R(t) ≈ 1
        assert_allclose(rt[:, 0], 1.0, atol=1e-3)

    def test_baseline_near_one_after_season(self):
        """Far after the active season ends, R(t) should return to 1.0."""
        step_dt = 7
        # start=0, dt_end=70 → season ends at day 70 (step 10)
        # Step 20 (day 140) should be well past the season
        params = make_envelope_params(start=0.0, dt_end=70.0, w_env=7.0, n=3)
        rt = EnvelopedLogisticRT().generate(params, num_time_steps=25, step_dt=step_dt)
        # After season, envelope E(t) ≈ 0, so R(t) ≈ 1
        assert_allclose(rt[:, 20], 1.0, atol=1e-3)

    def test_envelope_rises_at_start(self):
        """R(t) should rise from baseline as envelope activates."""
        step_dt = 7
        # start=70, dt_center very late so core stays near r_high
        params = make_envelope_params(
            start=70.0, dt_center=500.0, dt_end=500.0,
            w_env=7.0, r_low=0.5, r_high=2.5, n=2
        )
        rt = EnvelopedLogisticRT().generate(params, num_time_steps=30, step_dt=step_dt)
        # Well before start (step 5, day 35, ~35 days < 70): near 1.0
        assert_allclose(rt[:, 5], 1.0, atol=0.02)
        # At start inflection (step 10, day 70): envelope ≈ 0.5
        # C(t) ≈ r_high = 2.5, so R ≈ 1 + 0.5 * (2.5 - 1) = 1.75
        assert_allclose(rt[:, 10], 1.75, atol=0.1)
        # Well into season (step 15, day 105 >> 70): envelope ≈ 1
        # R ≈ 1 + 1 * (2.5 - 1) = 2.5
        assert_allclose(rt[:, 15], 2.5, atol=0.02)


class TestEnvelopedCoreLogistic:

    def test_core_decreases_over_time(self):
        """Inside the active season, R(t) should decrease as core logistic declines."""
        step_dt = 7
        # Season: start=0, dt_center=70, dt_end=200
        # At dt_center (day 70, step 10), core logistic should be at midpoint
        params = make_envelope_params(
            start=0.0, dt_center=70.0, dt_end=200.0,
            w_center=14.0, w_env=3.0,  # narrow envelope for quick activation
            r_low=0.5, r_high=2.5, n=3
        )
        rt = EnvelopedLogisticRT().generate(params, num_time_steps=40, step_dt=step_dt)

        # Early in season (step 5, day 35): core near r_high
        # Envelope already activated, so R should be high
        early_vals = rt[:, 5]

        # At center (step 10, day 70): core at midpoint (r_high + r_low)/2 = 1.5
        # Envelope fully activated, so R ≈ 1 + 1 * (1.5 - 1) = 1.5
        center_vals = rt[:, 10]

        # Late in season (step 20, day 140): core near r_low
        # Envelope still active, so R should be lower
        late_vals = rt[:, 20]

        # Check monotonic decrease
        assert np.all(early_vals > center_vals)
        assert np.all(center_vals > late_vals)

    def test_at_dt_center_envelope_modulates_midpoint(self):
        """At t = start + dt_center, with full envelope, core is at (r_high + r_low)/2."""
        step_dt = 7
        dt_center = 70.0  # 10 steps after start
        r_low, r_high = 0.4, 2.4
        # start=0, so center is at day 70 (step 10)
        # Use narrow envelope width and make dt_end large to keep envelope ≈ 1 at center
        params = make_envelope_params(
            start=0.0, dt_center=dt_center, dt_end=200.0,
            w_center=14.0, w_env=3.0,
            r_low=r_low, r_high=r_high, n=2
        )
        rt = EnvelopedLogisticRT().generate(params, num_time_steps=30, step_dt=step_dt)

        # At step 10 (day 70), envelope should be ≈ 1
        # Core = (r_high + r_low)/2 = 1.4
        # R(t) = 1 + 1 * (1.4 - 1) = 1.4
        expected_midpoint = (r_high + r_low) / 2.0
        assert_allclose(rt[:, 10], expected_midpoint, atol=0.05)


class TestEnvelopedPerSimulationVariation:

    def test_different_r_high_per_row(self):
        """Each simulation should use its own r_high parameter."""
        step_dt = 7
        params = pd.DataFrame(
            {
                "rt_logist_start":     [0.0, 0.0, 0.0],
                "rt_logist_dt_center": [500.0, 500.0, 500.0],  # far away
                "rt_logist_dt_end":    [200.0, 200.0, 200.0],
                "rt_logist_w_center":  [14.0, 14.0, 14.0],
                "rt_logist_w_env":     [3.0, 3.0, 3.0],
                "rt_logist_r_low":     [0.5, 0.5, 0.5],
                "rt_logist_r_high":    [2.0, 2.5, 3.0],
            }
        )
        rt = EnvelopedLogisticRT().generate(params, num_time_steps=20, step_dt=step_dt)

        # Mid-season (step 10, day 70), envelope ≈ 1, core ≈ r_high
        # R ≈ 1 + 1 * (r_high - 1)
        assert_allclose(rt[0, 10], 2.0, atol=0.05)
        assert_allclose(rt[1, 10], 2.5, atol=0.05)
        assert_allclose(rt[2, 10], 3.0, atol=0.05)

    def test_different_dt_end_per_row(self):
        """Different dt_end should cause envelope to close at different times."""
        step_dt = 7
        params = pd.DataFrame(
            {
                "rt_logist_start":     [0.0, 0.0],
                "rt_logist_dt_center": [500.0, 500.0],  # far away, core stays high
                "rt_logist_dt_end":    [70.0, 140.0],   # ends at day 70 vs 140
                "rt_logist_w_center":  [14.0, 14.0],
                "rt_logist_w_env":     [7.0, 7.0],
                "rt_logist_r_low":     [0.5, 0.5],
                "rt_logist_r_high":    [2.5, 2.5],
            }
        )
        rt = EnvelopedLogisticRT().generate(params, num_time_steps=30, step_dt=step_dt)

        # Step 10 is day 70
        # Sim 0: envelope falling at day 70 (end inflection), R dropping toward 1
        # Sim 1: envelope still fully open at day 70, R near core value

        # Step 15 is day 105
        # Sim 0: well past end (70), envelope ≈ 0, R ≈ 1
        assert_allclose(rt[0, 15], 1.0, atol=0.1)
        # Sim 1: still in season (< 140), R > 1
        assert rt[1, 15] > 1.5


class TestEnvelopedLogisticStaticMethod:

    def test_logistic_at_center_is_half(self):
        """At t=center, logistic function should return 0.5."""
        result = EnvelopedLogisticRT.logistic(t=100.0, center=100.0, width=10.0)
        assert_allclose(result, 0.5, rtol=1e-10)

    def test_logistic_far_before_center_near_zero(self):
        """Far before center, logistic should be near 0."""
        result = EnvelopedLogisticRT.logistic(t=0.0, center=100.0, width=10.0)
        assert result < 0.01

    def test_logistic_far_after_center_near_one(self):
        """Far after center, logistic should be near 1."""
        result = EnvelopedLogisticRT.logistic(t=200.0, center=100.0, width=10.0)
        assert result > 0.99

    def test_logistic_width_affects_steepness(self):
        """Smaller width should give steeper transition."""
        narrow = EnvelopedLogisticRT.logistic(t=110.0, center=100.0, width=2.0)
        wide = EnvelopedLogisticRT.logistic(t=110.0, center=100.0, width=20.0)
        # At t=110 (10 units past center):
        # narrow should be closer to 1, wide should be closer to 0.5
        assert narrow > wide


