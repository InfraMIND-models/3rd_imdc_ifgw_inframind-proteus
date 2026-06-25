from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats
from numpy.random import Generator


@dataclass
class InitialInfectionsConfig:
    """Initial infection seeding configuration.

    Attributes
    ----------
    method:
        Initialization method. Currently only ``"ones"`` is supported.
    params:
        Method-specific parameters.
    """
    method: str = "ones"
    num_steps: int = 7
    params: dict[str, float] = field(default_factory=dict)


def parse_initial_infections_config(config_dict: dict) -> InitialInfectionsConfig:
    """Parse initial infection seeding settings from a YAML-like config dict."""
    init_cfg = config_dict.get("initial_infections", {}) or {}

    method = str(init_cfg.get("method", "ones")).strip().lower()
    if method != "ones":
        raise ValueError(
            f"Unsupported initial_infections.method {method!r}. "
            "Supported methods: ['ones']"
        )

    num_steps = init_cfg.get("num_steps", InitialInfectionsConfig.num_steps)
    if num_steps <= 0:
        raise ValueError(
            f"initial_infections.num_steps must be > 0, got {num_steps}"
        )

    params_raw = init_cfg.get("params", {}) or {}
    if not isinstance(params_raw, dict):
        raise ValueError("initial_infections.params must be a dictionary")

    params = {str(k): float(v) for k, v in params_raw.items()}

    return InitialInfectionsConfig(method=method, params=params)


def build_initial_infec_df(
    num_simulations: int,
    gt_max_steps: int,  # UNUSED (deprecated in favor of initial_config.num_steps)
    step_dt: int,
    initial_config: InitialInfectionsConfig,
) -> pd.DataFrame:
    """Build the warm-up infection matrix used to seed the renewal loop.

    Returns a DataFrame of shape ``(num_simulations, gt_max_steps)``.
    Columns represent warm-up timestamps in days.
    """
    if num_simulations <= 0:
        raise ValueError(
            f"num_simulations must be > 0, got {num_simulations}"
        )
    # if gt_max_steps <= 0:
    #     raise ValueError(
    #         f"gt_max_steps must be > 0, got {gt_max_steps}"
    #     )
    if step_dt <= 0:
        raise ValueError(f"step_dt must be > 0, got {step_dt}")

    method = initial_config.method
    if method != "ones":
        raise ValueError(
            "build_initial_infec_df currently supports only method 'ones'"
        )

    # warmup_cols = list(range(0, gt_max_steps * step_dt, step_dt))  # Old: Based on generation time
    num_steps = initial_config.num_steps  # New: Independent parameter
    warmup_cols = list(range(0, num_steps * step_dt, step_dt))
    initial_infec_df = pd.DataFrame(
        np.ones((num_simulations, num_steps), dtype=float),
        columns=warmup_cols,
    )
    initial_infec_df.index.name = "i_simulation"
    initial_infec_df.columns.name = "t"

    return initial_infec_df
