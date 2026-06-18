"""
Calibration procedure of the outbreak dynamics model (renewal equation) of
Inframind Proteus.

This script is originally developed for the validation round of the
3rd Infodengue-Mosqlimate Dengue Challenge (3rd IMDC).

Usage
-----

"""

from __future__ import annotations

import io
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pydantic
import yaml.parser

from inframind_proteus import BaseConfig
from inframind_proteus.outbreak_dynamics import SimulationConfig
from inframind_proteus.outbreak_dynamics.utils import parse_set_arguments_with_yaml, load_yaml_dict


class Stage1Config(BaseConfig):
    """Stage 1 configuration: Broad exploration most free
    parameters."""

    num_simulations: int = 2**12


class Stage2Config(BaseConfig):
    """Stage 2 configuration: Focused exploration of a subset of
    free parameters.

    Nuisance parameters are fixed to optimal values
    provided by stage 1.
    """
    num_simulations: int = 2 ** 12

    free_params: list[str] =[
        "rt_logist_r_high",
        "rt_logist_start",
        # "notif_nb_overdispersion",
        # "notif_relative_scale",
    ]
    ll_temperature: float = 1.  # Higher values flatten the likelihood distribution
    rel_weight_cutoff: float = 1e-3  # Cutoff relative to maximum weight

    # Posterior building
    sampling_seed: int = 42  # Seed for any sampling procedure in stage 2 (e.g. KDE sampling)
    min_samples_to_kde: int = 1000  # Minimum number of samples to keep after cutoff (overrides cutoff if not met)
    max_samples_to_kde: int = 5000  # Maximum number of samples to keep, avoids heavy KDE calculation


class Stage3Config(BaseConfig):
    """Stage 3 configuration: Adjustment of confidence
    intervals to match coverages.
    """
    stage3_num_simulations: int = 2 ** 12


class ProgramConfig(BaseConfig):
    """Internal configuration data class for the calibration procedure script."""
    config_fpath: Path = Path("configs/calibrate_3rd_imdc_default.yaml")
    sim_config_fpath: Optional[Path] = None  # Optional path to a separate simulation config file (overrides config.sim_cfg)

    sim_cfg: SimulationConfig = SimulationConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()


def parse_args_get_dict(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse command-line arguments and return them as a dictionary."""
    parser = ArgumentParser()


    # --- Config file path
    parser.add_argument(
        "--config-fpath", "--cfg", "-c",
        default=ProgramConfig.config_fpath,
        type=Path,
        help="Path to the calibration YAML configuration file.",
    )

    # --- Generic nested `--set` argument.
    parser.add_argument(
        "--set",
        nargs=2,
        action="append",
        default=list(),
        metavar=("KEY", "VALUE"),
        help="Set a configuration parameter using dot notation (e.g. "
             "--set stage1.num_simulations 4096)."
             " Can be used multiple times.",
    )


    # ======

    args = parser.parse_args(argv)
    # Retain only informed arguments to avoid overriding config.
    args_dict = {k: v for k, v in args.__dict__.items() if v is not None}

    # Proceess --set arguments to override config values.
    set_args = args_dict.pop("set")
    overrides = parse_set_arguments_with_yaml(set_args)
    args_dict.update(overrides)

    return args_dict


class ProgramData:
    """Internal payload data class for the calibration procedure script."""


def main(argv: list[str] | None = None) -> None:

    # TEst
    argv += "--set sim_cfg.location.location_id RIOCLAROOO".split()

    # Program initialization sequence
    args_dict = parse_args_get_dict(argv)
    config_dict = load_yaml_dict(args_dict["config_fpath"])
    cfg = ProgramConfig()
    cfg.update_from_dict(config_dict)
    cfg.update_from_dict(args_dict)
    cfg.preprocess()




    pass

    cfg.preprocess()


if __name__ == "__main__":
    main(sys.argv[1:])
