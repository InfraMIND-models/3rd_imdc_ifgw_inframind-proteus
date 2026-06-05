"""Unit tests for inframind_proteus.outbreak_dynamics.sampling."""

import numpy as np
import pandas as pd

from inframind_proteus.outbreak_dynamics.sampling import sample_lhs


class TestSampleLhs:
    RANGES = {
        "alpha": [0.0, 1.0],
        "beta": [10.0, 100.0],
        "gamma": [-5.0, 5.0],
    }

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        df = sample_lhs(self.RANGES, num_simulations=50, rng=rng)
        assert df.shape == (50, 3)

    def test_column_names(self):
        rng = np.random.default_rng(0)
        df = sample_lhs(self.RANGES, num_simulations=10, rng=rng)
        assert list(df.columns) == list(self.RANGES.keys())

    def test_values_within_bounds(self):
        rng = np.random.default_rng(42)
        df = sample_lhs(self.RANGES, num_simulations=200, rng=rng)
        for col, (lo, hi) in self.RANGES.items():
            assert (df[col] >= lo).all(), f"{col}: value below lower bound"
            assert (df[col] <= hi).all(), f"{col}: value above upper bound"

    def test_single_simulation(self):
        rng = np.random.default_rng(0)
        df = sample_lhs({"x": [0.0, 1.0]}, num_simulations=1, rng=rng)
        assert df.shape == (1, 1)

    def test_reproducible_with_same_seed(self):
        ranges = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
        df1 = sample_lhs(ranges, num_simulations=20, rng=np.random.default_rng(7))
        df2 = sample_lhs(ranges, num_simulations=20, rng=np.random.default_rng(7))
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        ranges = {"x": [0.0, 1.0]}
        df1 = sample_lhs(ranges, num_simulations=20, rng=np.random.default_rng(1))
        df2 = sample_lhs(ranges, num_simulations=20, rng=np.random.default_rng(2))
        assert not df1.equals(df2)

    def test_lhs_stratification(self):
        """Each of N strata along each axis should contain exactly one sample."""
        n = 100
        rng = np.random.default_rng(0)
        df = sample_lhs({"x": [0.0, 1.0]}, num_simulations=n, rng=rng)
        strata = (df["x"] * n).astype(int).clip(0, n - 1)
        assert strata.nunique() == n, "LHS strata are not all covered"
