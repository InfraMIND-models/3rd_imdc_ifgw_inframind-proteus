"""Sampling helpers for the outbreak dynamics module.

This module groups parameter sampling utilities used to build simulation
ensembles. Additional sampling helpers should be added here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats
from numpy.random import Generator


@dataclass
class SamplingConfig:
    """Sampling strategy configuration.

    Attributes
    ----------
    method:
        Sampling method. Currently only ``"lhs"`` is supported.
    param_ranges:
        Parameter ranges to be sampled, mapping parameter name to
        ``[lower_bound, upper_bound]``.
    rt_params:
        Fixed reproduction-number parameters (from
        ``reproduction_number.params``).
    observation_params:
        Fixed observation-model parameters (from
        ``observation_model.params``).
    """

    # num_simulations: int# = 1000
    method: str = "lhs"
    param_ranges: dict[str, list[float]] = field(default_factory=dict)
    rt_params: dict[str, float] = field(default_factory=dict)
    observation_params: dict[str, float] = field(default_factory=dict)

def sample_lhs(
    param_ranges: dict[str, list[float]],
    num_simulations: int,
    rng: Generator,
) -> pd.DataFrame:
    """Draw a Latin Hypercube sample scaled to the given parameter ranges.

    Parameters
    ----------
    param_ranges:
        Mapping from parameter name to ``[lower_bound, upper_bound]``.
    num_simulations:
        Number of samples (rows in the output).
    rng:
        NumPy random generator instance.

    Returns
    -------
    pd.DataFrame
        Shape ``(num_simulations, len(param_ranges))``, columns are
        parameter names.
    """
    lhs_sampler = scipy.stats.qmc.LatinHypercube(d=len(param_ranges), rng=rng)
    lhs_samples = lhs_sampler.random(n=num_simulations)

    l_bounds = [v[0] for v in param_ranges.values()]
    u_bounds = [v[1] for v in param_ranges.values()]
    lhs_scaled = scipy.stats.qmc.scale(lhs_samples, l_bounds, u_bounds)

    return pd.DataFrame(lhs_scaled, columns=list(param_ranges.keys()))


def parse_calibration_sampling_config(config_dict: dict) -> SamplingConfig:
    """Parse calibration sampling settings from a YAML-like config dictionary.

    Expected sources:
    - ``simulation.num_simulations``
    - ``sampling.method`` and ``sampling.param_ranges``
    - ``reproduction_number.params``
    - ``observation_model.params``
    """
    sim_cfg = config_dict.get("simulation", {}) or {}
    sampling_cfg = config_dict.get("sampling", {}) or {}
    rt_cfg = config_dict.get("reproduction_number", {}) or {}
    obs_cfg = config_dict.get("observation_model", {}) or {}

    # num_simulations = int(sim_cfg.get("num_simulations", 1000))
    num_simulations = sim_cfg["num_simulations"]  # Required, no default
    if num_simulations <= 0:
        raise ValueError(
            f"simulation.num_simulations must be > 0, got {num_simulations}"
        )

    method = str(sampling_cfg.get("method", "lhs")).strip().lower()
    if method != "lhs":
        raise ValueError(
            f"Unsupported sampling.method {method!r}. Supported methods: ['lhs']"
        )

    param_ranges_raw = sampling_cfg.get("param_ranges", {}) or {}
    if not isinstance(param_ranges_raw, dict):
        raise ValueError("sampling.param_ranges must be a dictionary")

    param_ranges: dict[str, list[float]] = {}
    for name, bounds in param_ranges_raw.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"sampling.param_ranges[{name!r}] must be [lo, hi], got {bounds!r}"
            )
        lo = float(bounds[0])
        hi = float(bounds[1])
        if lo > hi:
            raise ValueError(
                f"sampling.param_ranges[{name!r}] has lo > hi: [{lo}, {hi}]"
            )
        param_ranges[str(name)] = [lo, hi]

    rt_params_raw = rt_cfg.get("params", {}) or {}
    obs_params_raw = obs_cfg.get("params", {}) or {}
    if not isinstance(rt_params_raw, dict):
        raise ValueError("reproduction_number.params must be a dictionary")
    if not isinstance(obs_params_raw, dict):
        raise ValueError("observation_model.params must be a dictionary")

    rt_params = {str(k): float(v) for k, v in rt_params_raw.items()}
    observation_params = {str(k): float(v) for k, v in obs_params_raw.items()}

    return SamplingConfig(
        # num_simulations=num_simulations,
        method=method,
        param_ranges=param_ranges,
        rt_params=rt_params,
        observation_params=observation_params,
    )


def build_calibration_params_df(
    num_simulations: int,
    sampling_config: SamplingConfig,
    required_param_names: list[str] | None = None,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """Build a calibration ``params_df`` from fixed and sampled parameters.

    Behavior:
    - Start from fixed RT and observation parameters.
    - Apply LHS sampling for keys listed in ``sampling.param_ranges``.
    - On name collision, sampled values override fixed values.
    """
    required_param_names = required_param_names or []

    # n = int(sampling_config.num_simulations)
    n = num_simulations
    if n <= 0:
        raise ValueError(f"num_simulations must be > 0, got {n}")

    fixed_params: dict[str, float] = {}
    fixed_params.update(sampling_config.rt_params)
    fixed_params.update(sampling_config.observation_params)

    if fixed_params:
        params_df = pd.DataFrame(
            {name: np.full(n, value, dtype=float) for name, value in fixed_params.items()}
        )
    else:
        params_df = pd.DataFrame(index=np.arange(n))

    param_ranges = sampling_config.param_ranges
    if param_ranges:
        if sampling_config.method != "lhs":
            raise ValueError(
                "build_calibration_params_df currently supports only sampling "
                "method 'lhs'"
            )
        rng = np.random.default_rng(rng_seed)
        sampled_df = sample_lhs(
            param_ranges=param_ranges,
            num_simulations=n,
            rng=rng,
        )
        for col in sampled_df.columns:
            params_df[col] = sampled_df[col].to_numpy()

    missing = [p for p in required_param_names if p not in params_df.columns]
    if missing:
        raise ValueError(
            "build_calibration_params_df produced params_df missing required "
            f"columns: {missing}. Provide them via reproduction_number.params, "
                "observation_model.params, or sampling.param_ranges."
        )

    return params_df
