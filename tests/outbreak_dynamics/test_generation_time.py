"""Unit tests for outbreak_dynamics.generation_time.

Design notes
------------
- GT parameters live on the model object, not in a params_df.
- Output shape is always (num_time_steps, gt_max_steps) — shared across all
  simulations.
- Axis 1 of the output is in reversed lag order: index 0 = largest lag.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from inframind_proteus.outbreak_dynamics.generation_time import (
    BaseGT,
    ConstantGammaGT,
)


# ---------------------------------------------------------------------------
# ConstantGammaGT — constructor
# ---------------------------------------------------------------------------

class TestConstantGammaGTConstructor:

    def test_stores_params(self):
        gt = ConstantGammaGT(shape=3.0, scale=5.0)
        assert gt.shape == 3.0
        assert gt.scale == 5.0

    def test_invalid_shape_zero(self):
        with pytest.raises(ValueError, match="shape"):
            ConstantGammaGT(shape=0.0, scale=5.0)

    def test_invalid_shape_negative(self):
        with pytest.raises(ValueError, match="shape"):
            ConstantGammaGT(shape=-1.0, scale=5.0)

    def test_invalid_scale_zero(self):
        with pytest.raises(ValueError, match="scale"):
            ConstantGammaGT(shape=3.0, scale=0.0)

    def test_invalid_scale_negative(self):
        with pytest.raises(ValueError, match="scale"):
            ConstantGammaGT(shape=3.0, scale=-2.0)


# ---------------------------------------------------------------------------
# ConstantGammaGT — get_pmf output shape and structure
# ---------------------------------------------------------------------------

class TestConstantGammaGTPmf:

    GT_MAX = 7
    NUM_TIME = 12
    STEP_DT = 7

    def _gt(self, shape=3.0, scale=5.0):
        return ConstantGammaGT(shape=shape, scale=scale)

    def test_output_shape(self):
        pmf = self._gt().get_pmf(self.GT_MAX, self.NUM_TIME, self.STEP_DT)
        assert pmf.shape == (self.NUM_TIME, self.GT_MAX)

    def test_output_shape_single_time_step(self):
        pmf = self._gt().get_pmf(gt_max_steps=5, num_time_steps=1, step_dt=7)
        assert pmf.shape == (1, 5)

    def test_all_values_non_negative(self):
        pmf = self._gt().get_pmf(self.GT_MAX, self.NUM_TIME, self.STEP_DT)
        assert np.all(pmf >= 0.0)

    def test_rows_sum_to_at_most_one(self):
        """PMF is truncated at gt_max_steps, so row sums should be ≤ 1."""
        pmf = self._gt().get_pmf(self.GT_MAX, self.NUM_TIME, self.STEP_DT)
        assert np.all(pmf.sum(axis=1) <= 1.0 + 1e-10)

    def test_rows_sum_close_to_one_for_large_window(self):
        """With a very large window, virtually all GT mass is captured."""
        pmf = self._gt(shape=3.0, scale=4.0).get_pmf(
            gt_max_steps=100, num_time_steps=5, step_dt=1
        )
        assert np.all(pmf.sum(axis=1) > 0.99)

    def test_all_time_rows_identical(self):
        """Constant GT — every row must be the same."""
        pmf = self._gt().get_pmf(self.GT_MAX, self.NUM_TIME, self.STEP_DT)
        for row in pmf:
            np.testing.assert_array_equal(row, pmf[0])

    def test_reversed_lag_order(self):
        """Index 0 should carry the largest-lag weight (smallest CDF mass
        near 0 days), and index -1 the smallest-lag weight.

        For a gamma with moderate shape and scale the mode is well above
        zero, so the PMF mass at small lags is lower than at larger lags.
        We verify the reversal by comparing against the un-reversed version.
        """
        gt = self._gt(shape=4.0, scale=5.0)   # mode = (shape-1)*scale = 15 days
        pmf = gt.get_pmf(gt_max_steps=10, num_time_steps=3, step_dt=7)
        row = pmf[0]

        # Reconstruct the natural (un-reversed) PMF via scipy
        gt_vals = np.arange(0, 11 * 7, 7)
        cdf = scipy.stats.gamma.cdf(gt_vals, a=4.0, scale=5.0)
        natural_pmf = np.diff(cdf)  # index 0 = smallest lag

        np.testing.assert_allclose(row, natural_pmf[::-1], rtol=1e-10)

    def test_larger_mean_shifts_mass(self):
        """A GT with a larger mean should have more mass at higher lags."""
        gt_fast = ConstantGammaGT(shape=2.0, scale=2.0)   # mean = 4 days
        gt_slow = ConstantGammaGT(shape=4.0, scale=7.0)   # mean = 28 days

        pmf_fast = gt_fast.get_pmf(gt_max_steps=10, num_time_steps=1, step_dt=7)[0]
        pmf_slow = gt_slow.get_pmf(gt_max_steps=10, num_time_steps=1, step_dt=7)[0]

        # In natural order (reversed back): fast GT has more mass at index 0
        # (small lag); slow GT has more mass at higher indices.
        # In our reversed representation: fast GT has more mass at the END,
        # slow GT has more mass at the START.
        assert pmf_slow[0] > pmf_fast[0]       # largest lag: slow > fast
        assert pmf_fast[-1] > pmf_slow[-1]     # smallest lag: fast > slow

    def test_different_step_dt(self):
        """Coarser step_dt should integrate more probability per step."""
        gt = self._gt(shape=3.0, scale=5.0)
        pmf_daily  = gt.get_pmf(gt_max_steps=10, num_time_steps=1, step_dt=1)[0]
        pmf_weekly = gt.get_pmf(gt_max_steps=10, num_time_steps=1, step_dt=7)[0]
        # Weekly bins integrate 7× as much probability → larger values
        assert pmf_weekly.sum() > pmf_daily.sum()

    def test_is_abstract_base(self):
        """BaseGT cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseGT()  # type: ignore[abstract]
