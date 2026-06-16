"""Sampling helpers for the outbreak dynamics module.

This module groups parameter sampling utilities used to build simulation
ensembles. Additional sampling helpers should be added here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats
from numpy.random import Generator


class PriorDistribution:

    def scale_from_unit_interval(self, x: np.ndarray) -> np.ndarray:
        """Scale a value from the unit interval [0, 1] to the distribution's support."""
        raise NotImplementedError("Must be implemented by subclasses")


class NormalPrior(PriorDistribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def scale_from_unit_interval(self, x: np.ndarray) -> np.ndarray:
        """Scale a value from the unit interval [0, 1] to the normal distribution's support."""
        return scipy.stats.norm.ppf(x, loc=self.mean, scale=self.std)


class GammaPrior(PriorDistribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        # Convert mean and std to shape and scale parameters for scipy's gamma distribution
        self.shape = (self.mean / self.std) ** 2
        self.scale = self.std ** 2 / self.mean

    def scale_from_unit_interval(self, x: np.ndarray) -> np.ndarray:
        """Scale a value from the unit interval [0, 1] to the gamma distribution's support."""
        return scipy.stats.gamma.ppf(x, a=self.shape, scale=self.scale)


def get_prior_distribution(prior_cfg: dict) -> PriorDistribution:

    dist_name = prior_cfg["distribution"]
    dist_params = prior_cfg["parameters"]

    cls_dict = {
        "normal": NormalPrior,
        "gamma": GammaPrior,
        # Add more distributions here as needed
    }

    try:
        cls = cls_dict[dist_name.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported distribution type: {dist_name!r}. "
            f"Supported distributions: {list(cls_dict.keys())}"
        )

    prior = cls(**dist_params)

    return prior



@dataclass
class SamplingConfig:
    """Sampling strategy configuration.

    Attributes
    ----------
    method:
        Sampling method. Currently only ``"lhs"`` is supported.
    rng_seed:
        Seed to the random number generator. If None, the generator will be
        initialized with a time-based seed.
    param_ranges:
        Parameter ranges to be sampled, mapping parameter name to
        ``[lower_bound, upper_bound]``.
    param_scales:
        Mapping from parameter name to scale type. Supported values:
        ``"linear"`` (default) or ``"log"`` (logarithmic).
    rt_params:
        Fixed reproduction-number parameters (from
        ``reproduction_number.params``).
    observation_params:
        Fixed observation-model parameters (from
        ``observation_model.params``).
    """

    # num_simulations: int# = 1000
    method: str = "lhs"
    rng_seed: int | None = None
    param_ranges: dict[str, list[float]] = field(default_factory=dict)
    param_scales: dict[str, str] = field(default_factory=dict)
    param_priors: dict[str, PriorDistribution] = field(default_factory=dict)
    rt_params: dict[str, float] = field(default_factory=dict)
    observation_params: dict[str, float] = field(default_factory=dict)


def _get_scaled_bounds(
    param_ranges: dict[str, list[float]],
    param_scales: dict[str, str],
    param_names: list[str],
):
    """ Get lower and upper bounds for sampling, applying transformation for
    non-linear scale parameters.

    For log-scale parameters, the bounds are transformed to log-space for sampling.

    Parameters
    ----------
    param_ranges:
        Mapping from parameter name to [lower_bound, upper_bound].
    param_scales:
        Mapping from parameter name to scale type. Supported values: "linear" or "log".
    param_names:
        List of parameter names to process (order matters for output).
    """
    l_bounds = []
    u_bounds = []

    for param_name in param_names:
        bounds = param_ranges[param_name]
        scale = param_scales.get(param_name, "linear").lower()

        if scale == "log":
            if bounds[0] <= 0 or bounds[1] <= 0:
                raise ValueError(
                    f"Log-scale parameter {param_name!r} must have positive bounds, "
                    f"got {bounds}"
                )
            l_bounds.append(np.log(bounds[0]))
            u_bounds.append(np.log(bounds[1]))

        elif scale == "linear":
            l_bounds.append(bounds[0])
            u_bounds.append(bounds[1])

        elif scale == "inverse":
            if bounds[0] * bounds[1] <= 0:
                raise ValueError(
                    f"Inverse-scale parameter {param_name!r} bounds must be both "
                    f"either positive or negative, excluding zero. "
                    f"got {bounds}"
                )
            l_bounds.append(1.0 / bounds[1])  # Inverse of upper bound becomes lower bound in transformed space
            u_bounds.append(1.0 / bounds[0])  # Inverse of lower bound becomes upper bound in transformed space

        else:
            raise ValueError(
                f"Unsupported scale for parameter {param_name!r}: {scale!r}. "
                "Supported scales: 'linear', 'log'"
            )

    return l_bounds, u_bounds



# def _transform_sampled_parameters(
#         lhs_scaled: np.ndarray,
#         param_scales: dict[str, str],
#         param_names: list[str],
# ):
#     """
#     Apply non-linear scaling transformations to the sampled parameters in-place.
#     """
#     for i, param_name in enumerate(param_names):
#         scale = param_scales.get(param_name, "linear").lower()
#
#         # Apply exponential transformation for log-scale parameters
#         if scale == "log":
#             lhs_scaled[:, i] = np.exp(lhs_scaled[:, i])
#
#         if scale == "inverse":
#             lhs_scaled[:, i] = 1.0 / lhs_scaled[:, i]


def _scale_params_with_priors(
    samples: np.ndarray,
    param_ranges: dict[str, list[float]],
    param_scales: dict[str, str],
    param_priors: dict[str, PriorDistribution],
    param_names: list[str],
) -> np.ndarray:
    """Scale samples from [0, 1] to parameter space using ranges, scales, or priors.

    For each parameter:
    - If a prior distribution is defined, use it to transform from [0, 1].
    - Otherwise, use param_ranges and param_scales (traditional bounds + scaling).

    Parameters
    ----------
    samples:
        Array of shape (n_samples, n_params) with values in [0, 1].
    param_ranges:
        Mapping from parameter name to [lower_bound, upper_bound].
    param_scales:
        Mapping from parameter name to scale type ('linear', 'log', 'inverse').
    param_priors:
        Mapping from parameter name to PriorDistribution instance.
    param_names:
        List of parameter names (order must match columns in samples).

    Returns
    -------
    np.ndarray
        Scaled samples in parameter space, same shape as input.
    """
    scaled = np.copy(samples)

    for i, param_name in enumerate(param_names):
        # Use prior distribution if available
        if param_name in param_priors:
            # Apply the prior PPF to scale from [0, 1] to the distribution's support
            prior = param_priors[param_name]
            scaled[:, i] = prior.scale_from_unit_interval(samples[:, i])

            # Priors are STILL bounded by param_ranges - they are clipped
            scaled[:, i] = np.clip(
                scaled[:, i],
                param_ranges[param_name][0],
                param_ranges[param_name][1]
            )

        # Otherwise use bounds and scale transformation
        else:
            bounds = param_ranges[param_name]
            scale = param_scales.get(param_name, "linear").lower()

            if scale == "log":
                if bounds[0] <= 0 or bounds[1] <= 0:
                    raise ValueError(
                        f"Log-scale parameter {param_name!r} must have positive bounds, "
                        f"got {bounds}"
                    )
                # Scale in log space then exponentiate
                log_lo = np.log(bounds[0])
                log_hi = np.log(bounds[1])
                scaled[:, i] = np.exp(samples[:, i] * (log_hi - log_lo) + log_lo)

            elif scale == "inverse":
                if bounds[0] * bounds[1] <= 0:
                    raise ValueError(
                        f"Inverse-scale parameter {param_name!r} bounds must be both "
                        f"positive or negative, excluding zero, got {bounds}"
                    )
                # Scale in inverse space then invert back
                inv_lo = 1.0 / bounds[1]  # Note: inverse swaps order
                inv_hi = 1.0 / bounds[0]
                scaled[:, i] = 1.0 / (samples[:, i] * (inv_hi - inv_lo) + inv_lo)

            elif scale == "linear":
                # Standard linear scaling
                scaled[:, i] = samples[:, i] * (bounds[1] - bounds[0]) + bounds[0]

            else:
                raise ValueError(
                    f"Unsupported scale for parameter {param_name!r}: {scale!r}. "
                    "Supported scales: 'linear', 'log', 'inverse'"
                )

    return scaled


def sample_lhs(
    param_ranges: dict[str, list[float]],
    num_simulations: int,
    rng: Generator,
    param_scales: dict[str, str] | None = None,
    param_priors: dict[str, PriorDistribution] | None = None,
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
    param_scales:
        Mapping from parameter name to scale type. Supported values:
        ``"linear"`` (default), ``"log"`` (logarithmic), or ``"inverse"``.
        For log-scale parameters, the bounds are interpreted as linear values,
        but sampling occurs in log-space: samples are drawn uniformly from
        ``[log(lower), log(upper)]`` and then exponentiated.
    param_priors:
        Mapping from parameter name to PriorDistribution instance.
        If provided for a parameter, it takes precedence over param_scales.

    Returns
    -------
    pd.DataFrame
        Shape ``(num_simulations, len(param_ranges))``, columns are
        parameter names.
    """
    param_scales = param_scales or {}
    param_priors = param_priors or {}
    param_names = list(param_ranges.keys())

    lhs_sampler = scipy.stats.qmc.LatinHypercube(d=len(param_ranges), rng=rng)
    lhs_samples = lhs_sampler.random(n=num_simulations)

    # Scale from [0, 1] to parameter space using priors or bounds+scales
    lhs_scaled = _scale_params_with_priors(
        samples=lhs_samples,
        param_ranges=param_ranges,
        param_scales=param_scales,
        param_priors=param_priors,
        param_names=param_names,
    )


    return pd.DataFrame(lhs_scaled, columns=param_names)


def sample_sobol(
    param_ranges: dict[str, list[float]],
    num_simulations: int,
    rng: Generator,
    param_scales: dict[str, str] | None = None,
    param_priors: dict[str, PriorDistribution] | None = None,
) -> pd.DataFrame:
    """Draw a Sobol sequence sample scaled to the given parameter ranges.

    Parameters
    ----------
    param_ranges:
        Mapping from parameter name to ``[lower_bound, upper_bound]``.
    num_simulations:
        Number of samples (rows in the output).
    rng:
        NumPy random generator instance.
    param_scales:
        Mapping from parameter name to scale type. Supported values:
        ``"linear"`` (default), ``"log"`` (logarithmic), or ``"inverse"``.
        For log-scale parameters, the bounds are interpreted as linear values,
        but sampling occurs in log-space: samples are drawn uniformly from
        ``[log(lower), log(upper)]`` and then exponentiated.
    param_priors:
        Mapping from parameter name to PriorDistribution instance.
        If provided for a parameter, it takes precedence over param_scales.

    Returns
    -------
    pd.DataFrame
        Shape ``(num_simulations, len(param_ranges))``, columns are
        parameter names.
    """
    param_scales = param_scales or {}
    param_priors = param_priors or {}
    param_names = list(param_ranges.keys())

    sampler = scipy.stats.qmc.Sobol(d=len(param_ranges), rng=rng)
    samples = sampler.random(n=num_simulations)

    # Scale from [0, 1] to parameter space using priors or bounds+scales
    scaled_samples = _scale_params_with_priors(
        samples=samples,
        param_ranges=param_ranges,
        param_scales=param_scales,
        param_priors=param_priors,
        param_names=param_names,
    )


    return pd.DataFrame(scaled_samples, columns=param_names)


def parse_calibration_sampling_config(config_dict: dict) -> SamplingConfig:
    """Parse calibration sampling settings from a YAML-like config dictionary.

    Expected sources:
    - ``simulation.num_simulations``
    - ``sampling.method``, ``sampling.param_ranges``, and ``sampling.scale``
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

    rng_seed = sampling_cfg.get("rng_seed", None)
    if not isinstance(rng_seed, int) and rng_seed is not None:
        raise TypeError(f"sampling.rng_seed must be an integer or None, got {rng_seed!r}")
    if rng_seed is not None and rng_seed < 0:
        raise ValueError(f"sampling.rng_seed must be non-negative, got {rng_seed}")

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

    # Parse parameter scales (default to "linear")
    param_scales_raw = sampling_cfg.get("param_scales", {}) or {}
    if not isinstance(param_scales_raw, dict):
        raise ValueError("sampling.scale must be a dictionary")
    
    param_scales: dict[str, str] = {}
    for name, scale_type in param_scales_raw.items():
        scale_type_str = str(scale_type).strip().lower()
        if scale_type_str not in ("linear", "log"):
            raise ValueError(
                f"sampling.scale[{name!r}] must be 'linear' or 'log', "
                f"got {scale_type_str!r}"
            )
        param_scales[str(name)] = scale_type_str

    param_priors: dict[str, PriorDistribution] = {}
    priors_cfg_dict = sampling_cfg.get("param_priors", {}) or {}
    if priors_cfg_dict is None:
        priors_cfg_dict = {}
    if not isinstance(priors_cfg_dict, dict):
        raise ValueError("sampling.param_priors must be a dictionary")
    for name, prior_cfg in priors_cfg_dict.items():
        if not isinstance(prior_cfg, dict):
            raise ValueError(f"sampling.param_priors[{name!r}] must be a dictionary, got {prior_cfg!r}")

        if "distribution" not in prior_cfg:
            raise ValueError(f"sampling.param_priors[{name!r}] missing required 'distribution' key")
        if "parameters" not in prior_cfg:
            raise ValueError(f"sampling.param_priors[{name!r}] missing required 'parameters' key")

        param_priors[str(name)] = get_prior_distribution(prior_cfg)


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
        rng_seed=rng_seed,
        param_ranges=param_ranges,
        param_scales=param_scales,
        param_priors=param_priors,
        rt_params=rt_params,
        observation_params=observation_params,
    )


def build_calibration_params_df(
    num_simulations: int,
    sampling_config: SamplingConfig,
    required_param_names: list[str] | None = None,
    **kwargs
) -> pd.DataFrame:
    """Build a calibration ``params_df`` from fixed and sampled parameters.

    Behavior:
    - Start from fixed RT and observation parameters.
    - Apply LHS sampling for keys listed in ``sampling.param_ranges``.
    - On name collision, sampled values override fixed values.
    - Parameter scales from ``sampling.scale`` are applied during sampling.
    """
    required_param_names = required_param_names or []

    # n = int(sampling_config.num_simulations)
    if num_simulations<= 0:
        raise ValueError(f"num_simulations must be > 0, got {num_simulations}")

    fixed_params: dict[str, float] = {}
    fixed_params.update(sampling_config.rt_params)
    fixed_params.update(sampling_config.observation_params)

    param_ranges = sampling_config.param_ranges

    if param_ranges:

        # rng = np.random.default_rng(rng_seed)
        rng = np.random.default_rng(sampling_config.rng_seed)

        if sampling_config.method == "lhs":
            sampled_df = sample_lhs(
                param_ranges=param_ranges,
                num_simulations=num_simulations,
                rng=rng,
                param_scales=sampling_config.param_scales,
                param_priors=sampling_config.param_priors,
            )

        elif sampling_config.method == "sobol":
            sampled_df = sample_sobol(
                param_ranges=param_ranges,
                num_simulations=num_simulations,
                rng=rng,
                param_scales=sampling_config.param_scales,
                param_priors=sampling_config.param_priors,
            )

        elif sampling_config.method == "given":
            # Build with pre-sampled parameters provided via kwargs
            if "given_params" not in kwargs:
                raise ValueError(
                    "sampling.method is 'given' but no 'given_params' provided in kwargs"
                )
            sampled_df = kwargs["given_params"]

        else:
            raise ValueError(
                f"Unsupported sampling.method: {sampling_config.method!r}. "
            )

        # ---
        # Check for sample size change (expected for some methods)
        if sampled_df.shape[0] != num_simulations:
            warnings.warn(
                f"Sampling method \"{sampling_config.method!r}\" produced "
                f"{len(sampled_df)} samples, different from {num_simulations=}."
                f" This is expected for some methods, "
                f"but you may have to update num_simulations to `params_df.shape[0]`"
                f" for the new value after this function."
            )
            num_simulations =  sampled_df.shape[0]

    else:
        sampled_df = pd.DataFrame({})

    # Combine fixed and sampled
    if fixed_params:
        params_df = pd.DataFrame(
            {name: np.full(num_simulations, value, dtype=float) for name, value in fixed_params.items()}
        )
    else:
        params_df = pd.DataFrame(index=np.arange(num_simulations))
    params_df.index.name = "i_simulation"

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
