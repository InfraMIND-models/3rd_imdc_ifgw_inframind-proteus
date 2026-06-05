"""Outbreak dynamics simulation module for inframind-proteus.

Public API
----------
RenewalSimulator
    Vectorised renewal equation simulator.

SimulationConfig, TemporalConfig, LocationConfig
    Configuration dataclasses.

SimulationOutput
    Container for simulation results.

LogisticRT
    Logistic R(t) model (first implemented R(t) form).

ConstantGammaGT
    Constant gamma generation time PMF model.

wis_score_vectorized
    Weighted Interval Score over case beam quantiles.
"""

from .generation_time import BaseGT, ConstantGammaGT
from .initial_infections import (
    InitialInfectionsConfig,
    build_initial_infec_df,
    parse_initial_infections_config,
)
from .rt_models import BaseRT, LogisticRT
from .sampling import (
    SamplingConfig,
    build_calibration_params_df,
    parse_calibration_sampling_config,
    sample_lhs,
)
from .scoring import nbinom_ppf_cf, wis_score_vectorized
from .simulator import (
    LocationConfig,
    RenewalSimulator,
    SimulationConfig,
    SimulationOutput,
    TemporalConfig,
)

__all__ = [
    "BaseGT",
    "BaseRT",
    "ConstantGammaGT",
    "InitialInfectionsConfig",
    "LocationConfig",
    "LogisticRT",
    "RenewalSimulator",
    "SamplingConfig",
    "SimulationConfig",
    "SimulationOutput",
    "TemporalConfig",
    "build_calibration_params_df",
    "build_initial_infec_df",
    "nbinom_ppf_cf",
    "parse_calibration_sampling_config",
    "parse_initial_infections_config",
    "sample_lhs",
    "wis_score_vectorized",
]
