"""Unit tests for inframind_proteus.outbreak_dynamics.sampling."""

import numpy as np
import pandas as pd
import pytest

from inframind_proteus.outbreak_dynamics.sampling import (
    SamplingConfig,
    build_calibration_params_df,
    parse_calibration_sampling_config,
    sample_lhs,
)


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


class TestParseCalibrationSamplingConfig:
    def test_parse_reads_sampling_and_fixed_sections(self):
        config_dict = {
            "simulation": {"num_simulations": 10},
            "sampling": {
                "method": "lhs",
                "param_ranges": {
                    "rt_logist_width": [1, 50],
                    "notif_scaling_factor": [10, 1000],
                },
            },
            "reproduction_number": {
                "params": {
                    "rt_logist_roff": 1.0,
                    "rt_logist_start": 50.0,
                }
            },
            "observation_model": {
                "params": {
                    "notif_nb_overdispersion": 10.0,
                    "notif_scaling_factor": 1.0,
                }
            },
        }

        parsed = parse_calibration_sampling_config(config_dict)
        assert parsed.method == "lhs"
        assert parsed.param_ranges["rt_logist_width"] == [1.0, 50.0]
        assert parsed.rt_params["rt_logist_roff"] == pytest.approx(1.0)
        assert parsed.observation_params["notif_nb_overdispersion"] == pytest.approx(10.0)

    def test_parse_defaults_to_lhs_with_empty_ranges(self):
        parsed = parse_calibration_sampling_config(
            {"simulation": {"num_simulations": 10}}
        )
        assert parsed.method == "lhs"
        assert parsed.param_ranges == {}
        assert parsed.rt_params == {}
        assert parsed.observation_params == {}

    def test_parse_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unsupported sampling.method"):
            parse_calibration_sampling_config(
                {
                    "simulation": {"num_simulations": 10},
                    "sampling": {"method": "sobol", "param_ranges": {}},
                }
            )

    def test_parse_rejects_bad_range_shape(self):
        with pytest.raises(ValueError, match=r"must be \[lo, hi\]"):
            parse_calibration_sampling_config(
                {
                    "simulation": {"num_simulations": 10},
                    "sampling": {"param_ranges": {"x": [0.0, 1.0, 2.0]}},
                }
            )

    def test_parse_rejects_reversed_range(self):
        with pytest.raises(ValueError, match="lo > hi"):
            parse_calibration_sampling_config(
                {
                    "simulation": {"num_simulations": 10},
                    "sampling": {"param_ranges": {"x": [2.0, 1.0]}},
                }
            )


class TestBuildCalibrationParamsDf:
    def test_fixed_only_returns_repeated_rows(self):
        cfg = SamplingConfig(
            method="lhs",
            param_ranges={},
            rt_params={"rt_logist_roff": 1.0},
            observation_params={"notif_nb_overdispersion": 8.0},
        )
        out = build_calibration_params_df(
            num_simulations=4,
            sampling_config=cfg,
            required_param_names=["rt_logist_roff", "notif_nb_overdispersion"],
        )
        assert out.shape == (4, 2)
        assert (out["rt_logist_roff"] == 1.0).all()
        assert (out["notif_nb_overdispersion"] == 8.0).all()

    def test_lhs_overrides_fixed_values_for_same_column(self):
        cfg = SamplingConfig(
            method="lhs",
            param_ranges={"notif_scaling_factor": [10.0, 20.0]},
            rt_params={},
            observation_params={"notif_scaling_factor": 1.0},
        )
        out = build_calibration_params_df(
            num_simulations=20,
            sampling_config=cfg,
            required_param_names=["notif_scaling_factor"],
            rng_seed=0,
        )
        assert out.shape == (20, 1)
        assert (out["notif_scaling_factor"] >= 10.0).all()
        assert (out["notif_scaling_factor"] <= 20.0).all()
        assert not (out["notif_scaling_factor"] == 1.0).all()

    def test_lhs_reproducible_with_same_seed(self):
        cfg = SamplingConfig(
            method="lhs",
            param_ranges={"x": [0.0, 1.0]},
            rt_params={},
            observation_params={},
        )
        out1 = build_calibration_params_df(
            num_simulations=25,
            sampling_config=cfg,
            required_param_names=["x"],
            rng_seed=7,
        )
        out2 = build_calibration_params_df(
            num_simulations=25,
            sampling_config=cfg,
            required_param_names=["x"],
            rng_seed=7,
        )
        pd.testing.assert_frame_equal(out1, out2)

    def test_missing_required_columns_raises(self):
        cfg = SamplingConfig(
            method="lhs",
            param_ranges={},
            rt_params={"rt_logist_roff": 1.0},
            observation_params={},
        )
        with pytest.raises(ValueError, match="missing required columns"):
            build_calibration_params_df(
                num_simulations=3,
                sampling_config=cfg,
                required_param_names=["rt_logist_roff", "notif_nb_overdispersion"],
            )


class TestSampleLhsWithScale:
    """Tests for log-scale and mixed-scale parameter sampling."""

    def test_log_scale_values_within_bounds(self):
        """Log-scale samples should be within specified bounds in linear space."""
        ranges = {"x": [1.0, 100.0]}
        param_scales = {"x": "log"}
        rng = np.random.default_rng(42)
        df = sample_lhs(ranges, num_simulations=200, rng=rng, param_scales=param_scales)
        assert (df["x"] >= 1.0).all(), "log-scale value below lower bound"
        assert (df["x"] <= 100.0).all(), "log-scale value above upper bound"

    def test_log_scale_stratification_in_log_space(self):
        """Each stratum in log-space should contain approximately one sample."""
        n = 100
        ranges = {"x": [1.0, 100.0]}
        param_scales = {"x": "log"}
        rng = np.random.default_rng(0)
        df = sample_lhs(ranges, num_simulations=n, rng=rng, param_scales=param_scales)
        # Transform to log space and check stratification
        log_x = np.log(df["x"])
        strata = ((log_x - np.log(1.0)) / (np.log(100.0) - np.log(1.0)) * n).astype(int).clip(0, n - 1)
        assert strata.nunique() >= n - 5, "log-scale strata not sufficiently covered"

    def test_mixed_linear_and_log_scale(self):
        """Mix linear and log-scale parameters."""
        ranges = {
            "linear_param": [0.0, 1.0],
            "log_param": [0.1, 100.0],
        }
        param_scales = {
            "linear_param": "linear",
            "log_param": "log",
        }
        rng = np.random.default_rng(42)
        df = sample_lhs(
            ranges,
            num_simulations=100,
            rng=rng,
            param_scales=param_scales,
        )
        # Check linear scale bounds
        assert (df["linear_param"] >= 0.0).all()
        assert (df["linear_param"] <= 1.0).all()
        # Check log scale bounds
        assert (df["log_param"] >= 0.1).all()
        assert (df["log_param"] <= 100.0).all()

    def test_log_scale_reproducible_with_same_seed(self):
        """Log-scale sampling should be reproducible with same seed."""
        ranges = {"x": [0.1, 10.0]}
        param_scales = {"x": "log"}
        df1 = sample_lhs(
            ranges,
            num_simulations=50,
            rng=np.random.default_rng(7),
            param_scales=param_scales,
        )
        df2 = sample_lhs(
            ranges,
            num_simulations=50,
            rng=np.random.default_rng(7),
            param_scales=param_scales,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_default_scale_is_linear(self):
        """Unspecified scale should default to linear."""
        ranges = {"x": [1.0, 10.0]}
        rng = np.random.default_rng(42)
        df = sample_lhs(ranges, num_simulations=100, rng=rng, param_scales={})
        # Linear sampling should have a relatively uniform distribution
        # (compared to log-scale which would cluster toward lower values)
        assert df["x"].std() > 2.0, "linear scale not producing expected distribution"

    def test_invalid_scale_type_raises(self):
        """Invalid scale type should raise ValueError."""
        ranges = {"x": [1.0, 10.0]}
        param_scales = {"x": "invalid"}
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Unsupported scale"):
            sample_lhs(ranges, num_simulations=10, rng=rng, param_scales=param_scales)

    def test_log_scale_with_negative_bounds_raises(self):
        """Log-scale with negative bounds should cause math domain error."""
        ranges = {"x": [-10.0, 10.0]}
        param_scales = {"x": "log"}
        rng = np.random.default_rng(0)
        with pytest.raises((ValueError)):
            sample_lhs(ranges, num_simulations=10, rng=rng, param_scales=param_scales)


class TestParseCalibrationSamplingConfigWithScale:
    """Tests for parsing scale configuration."""

    def test_parse_scale_section(self):
        """Parse scale configuration from config dict."""
        config_dict = {
            "simulation": {"num_simulations": 10},
            "sampling": {
                "method": "lhs",
                "param_ranges": {
                    "param_a": [1.0, 10.0],
                    "param_b": [100.0, 1000.0],
                },
                "param_scales": {
                    "param_b": "log",
                },
            },
        }
        parsed = parse_calibration_sampling_config(config_dict)
        assert parsed.param_scales["param_b"] == "log"
        # param_a not in scale dict, so it should not be in param_scales or default to linear
        assert parsed.param_scales.get("param_a", "linear") == "linear"

    def test_parse_empty_scale_section(self):
        """Empty or missing scale section should work fine."""
        config_dict = {
            "simulation": {"num_simulations": 10},
            "sampling": {
                "method": "lhs",
                "param_ranges": {"param_a": [1.0, 10.0]},
            },
        }
        parsed = parse_calibration_sampling_config(config_dict)
        assert parsed.param_scales == {}

    def test_parse_invalid_scale_type_raises(self):
        """Invalid scale type should raise ValueError."""
        config_dict = {
            "simulation": {"num_simulations": 10},
            "sampling": {
                "param_ranges": {"x": [1.0, 10.0]},
                "param_scales": {"x": "exponential"},
            },
        }
        with pytest.raises(ValueError, match="must be 'linear' or 'log'"):
            parse_calibration_sampling_config(config_dict)

    def test_parse_scale_case_insensitive(self):
        """Scale types should be case-insensitive."""
        config_dict = {
            "simulation": {"num_simulations": 10},
            "sampling": {
                "param_ranges": {
                    "param_a": [1.0, 10.0],
                    "param_b": [1.0, 10.0],
                },
                "param_scales": {
                    "param_a": "LOG",
                    "param_b": "LINEAR",
                },
            },
        }
        parsed = parse_calibration_sampling_config(config_dict)
        assert parsed.param_scales["param_a"] == "log"
        assert parsed.param_scales["param_b"] == "linear"


class TestBuildCalibrationParamsDfWithScale:
    """Tests for building params_df with scale support."""

    def test_lhs_with_log_scale_param(self):
        """Build params_df with log-scale sampling."""
        cfg = SamplingConfig(
            method="lhs",
            param_ranges={"x": [0.1, 100.0]},
            param_scales={"x": "log"},
            rt_params={},
            observation_params={},
        )
        out = build_calibration_params_df(
            num_simulations=50,
            sampling_config=cfg,
            required_param_names=["x"],
            rng_seed=0,
        )
        assert out.shape == (50, 1)
        assert (out["x"] >= 0.1).all()
        assert (out["x"] <= 100.0).all()

    def test_mixed_scale_params_in_build_df(self):
        """Build params_df with mixed linear and log scales."""
        cfg = SamplingConfig(
            method="lhs",
            param_ranges={
                "linear": [1.0, 10.0],
                "log": [0.1, 100.0],
            },
            param_scales={"log": "log"},
            rt_params={"fixed_param": 5.0},
            observation_params={},
        )
        out = build_calibration_params_df(
            num_simulations=50,
            sampling_config=cfg,
            required_param_names=["linear", "log", "fixed_param"],
            rng_seed=42,
        )
        assert out.shape == (50, 3)
        assert (out["linear"] >= 1.0).all()
        assert (out["linear"] <= 10.0).all()
        assert (out["log"] >= 0.1).all()
        assert (out["log"] <= 100.0).all()
        assert (out["fixed_param"] == 5.0).all()
