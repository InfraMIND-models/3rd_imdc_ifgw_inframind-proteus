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

from inframind_proteus.outbreak_dynamics.rt_models import BaseRT, LogisticRT


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
