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
