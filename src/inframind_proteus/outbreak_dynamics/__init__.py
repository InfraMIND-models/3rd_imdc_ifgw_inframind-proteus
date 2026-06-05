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
from .rt_models import BaseRT, LogisticRT
from .sampling import sample_lhs
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
    "LocationConfig",
    "LogisticRT",
    "RenewalSimulator",
    "SimulationConfig",
    "SimulationOutput",
    "TemporalConfig",
    "nbinom_ppf_cf",
    "sample_lhs",
    "wis_score_vectorized",
]
