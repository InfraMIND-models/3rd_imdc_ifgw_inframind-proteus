"""Unit tests for inframind_proteus.outbreak_dynamics.utils."""

import numpy as np
import pandas as pd
import pytest

from inframind_proteus.outbreak_dynamics.utils import (
    epiweek_to_date,
    parse_timestamp,
    sample_lhs,
)


# ---------------------------------------------------------------------------
# sample_lhs
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# epiweek_to_date
# ---------------------------------------------------------------------------

class TestEpiweekToDate:
    def test_returns_timestamp(self):
        result = epiweek_to_date(202340)
        assert isinstance(result, pd.Timestamp)

    def test_known_week_40_2023(self):
        # CDC epiweek 40 of 2023 opens on Sunday 2023-10-01
        assert epiweek_to_date(202340) == pd.Timestamp("2023-10-01")

    def test_known_week_1_2023(self):
        # CDC epiweek 1 of 2023 opens on Sunday 2023-01-01
        assert epiweek_to_date(202301) == pd.Timestamp("2023-01-01")

    def test_result_is_sunday(self):
        # CDC epiweeks always start on Sunday
        result = epiweek_to_date(202315)
        assert result.day_name() == "Sunday"

    def test_different_years(self):
        d2022 = epiweek_to_date(202201)
        d2024 = epiweek_to_date(202401)
        assert d2022.year == 2022
        # CDC week 1/2024 opens on 2023-12-31 (crosses year boundary — expected)
        assert epiweek_to_date(202401) == pd.Timestamp("2023-12-31")
        assert d2022 < d2024


# ---------------------------------------------------------------------------
# parse_timestamp
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    ZERO = pd.Timestamp("2023-10-01")

    def test_iso_string(self):
        result = parse_timestamp("2023-10-01")
        assert result == pd.Timestamp("2023-10-01")

    def test_iso_string_with_time(self):
        result = parse_timestamp("2023-10-01T00:00:00")
        assert result == pd.Timestamp("2023-10-01")

    def test_epiweek_integer(self):
        result = parse_timestamp(202340)
        assert result == epiweek_to_date(202340)

    def test_float_zero_offset(self):
        result = parse_timestamp(0.0, zero_date=self.ZERO)
        assert result == self.ZERO

    def test_float_one_week_offset(self):
        result = parse_timestamp(7.0, zero_date=self.ZERO)
        assert result == self.ZERO + pd.Timedelta(days=7)

    def test_float_requires_zero_date(self):
        with pytest.raises(ValueError, match="zero_date"):
            parse_timestamp(7.0)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            parse_timestamp(None)  # type: ignore[arg-type]

    def test_list_raises(self):
        with pytest.raises(TypeError):
            parse_timestamp([2023, 10, 1])  # type: ignore[arg-type]
