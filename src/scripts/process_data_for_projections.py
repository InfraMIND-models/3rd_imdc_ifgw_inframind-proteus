"""Prepare necessary data for projections with the outbreak dynamic model.

This script includes at least two crucial stages:
- Combine calibration distributions and predictions from multiple years.
- Incorporate predictions from the outbreak features model to generate parameter
  samples for the projections.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

print("Importing libraries...")
from inframind_proteus import BaseConfig
from inframind_proteus.outbreak_dynamics.utils import load_yaml_dict, parse_set_arguments_with_yaml, add_set_argument


def main(argv: Union[list[str], None] = None):
    """process_data_for_projections.py"""
    _initialize_program(argv)


class ProgramConfig(BaseConfig):
    """Internal configuration data class for the
    `process_data_for_projections` script.
    """
    config_fpath: Path = Path("configs/process_data_for_projections_default.yaml")
    base_sim_config_fpath: Path = Path("configs/simulation_config_default.yaml")
    output_dir: Path = Path("output/validation_round_projections")


class ProgramData:
    """Internal data class for the `process_data_for_projections` script.
    """


def parse_args_get_dict(argv) -> dict:
    """"""
    parser = ArgumentParser()

    # --- Config file path
    parser.add_argument(
        "--config-fpath", "--cfg", "-c",
        default=ProgramConfig.config_fpath,
        type=Path,
        help="Path to the calibration YAML configuration file.",
    )

    parser.add_argument(
        "--output-dir", "--out", "-o",
        default=ProgramConfig.output_dir,
        type=Path,
        help="Path to the output directory.",
    )

    # --- Generic nested `--set` argument.
    add_set_argument(parser)

    # ======

    args = parser.parse_args(argv)
    # Retain only informed arguments to avoid overriding config.
    args_dict = {k: v for k, v in args.__dict__.items() if v is not None}

    # Proceess --set arguments to override config values.
    set_args = args_dict.pop("set")
    overrides = parse_set_arguments_with_yaml(set_args)
    args_dict.update(overrides)

    return args_dict


def _initialize_program(argv):
    # --- Program initialization sequence
    args_dict = parse_args_get_dict(argv)
    cfg = ProgramConfig()
    if "config_fpath" in args_dict:
        config_dict = load_yaml_dict(args_dict["config_fpath"])
        cfg.update_from_dict(config_dict)
    cfg.update_from_dict(args_dict)
    cfg.preprocess()
    data = ProgramData()

    print(args_dict)


if __name__ == "__main__":
    main(sys.argv[1:])
