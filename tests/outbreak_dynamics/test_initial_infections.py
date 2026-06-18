"""Unit tests for inframind_proteus.outbreak_dynamics.initial_infections."""

import numpy as np
import pytest

from inframind_proteus.outbreak_dynamics.initial_infections import (
    InitialInfectionsConfig,
    build_initial_infec_df,
    parse_initial_infections_config,
)


class TestParseInitialInfectionsConfig:
    def test_defaults_to_ones(self):
        parsed = parse_initial_infections_config({})
        assert parsed.method == "ones"
        assert parsed.params == {}

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unsupported initial_infections.method"):
            parse_initial_infections_config({"initial_infections": {"method": "other"}})

    def test_rejects_bad_params_type(self):
        with pytest.raises(ValueError, match="initial_infections.params"):
            parse_initial_infections_config({"initial_infections": {"params": [1, 2]}})


class TestBuildInitialInfecDf:
    def test_builds_expected_shape_and_values(self):
        cfg = InitialInfectionsConfig(
            method="ones", params={}, num_steps=4
        )
        df = build_initial_infec_df(
            num_simulations=3,
            gt_max_steps=7,
            step_dt=7,
            initial_config=cfg,
        )
        assert df.shape == (3, 4)
        assert np.allclose(df.values, 1.0)
        assert list(df.columns) == [0, 7, 14, 21]

    def test_rejects_invalid_dimensions(self):
        cfg = InitialInfectionsConfig(method="ones", params={})
        with pytest.raises(ValueError, match="num_simulations"):
            build_initial_infec_df(0, 2, 7, cfg)
        # with pytest.raises(ValueError, match="gt_max_steps"):
        #     build_initial_infec_df(2, 0, 7, cfg)
        with pytest.raises(ValueError, match="step_dt"):
            build_initial_infec_df(2, 2, 0, cfg)
