#!/usr/bin/env python
"""CLI entry point for the inframind-proteus outbreak dynamics simulator.

DEPRECATION WARNING: This script was created as a stub during initial development,
but it was never used. The dynamic model can be run through `calibrate_3rd_imdc`
and `project_3rd_imdc` scripts instead.
This script may be removed in future releases.

Usage
-----
    uv run python src/scripts/run_outbreak_dynamics.py config.yaml

Override any YAML key using dot-notation::

    uv run python src/scripts/run_outbreak_dynamics.py config.yaml \\
        --set simulation.num_simulations=500 \\
        --set simulation.mode=calibration

Override specific important parameters directly::

    uv run python src/scripts/run_outbreak_dynamics.py config.yaml \\
        --mode calibration \\
        --num-simulations 500 \\
        --rng-seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from inframind_proteus.outbreak_dynamics import RenewalSimulator


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_outbreak_dynamics",
        description="Run the inframind-proteus outbreak dynamics simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML configuration file.",
    )

    # --- Generic dot-notation overrides
    parser.add_argument(
        "--set",
        metavar="KEY VALUE",
        action="append",
        dest="overrides",
        default=[],
        help=(
            "Override any YAML key using dot-notation. "
            "Repeatable: --set a.b 1 --set c 2"
        ),
    )

    # --- Named parameter shortcuts
    # TODO: extend this list when important parameters are agreed upon
    parser.add_argument(
        "--mode",
        choices=["calibration", "projection"],
        default=None,
        help="Simulation mode (overrides config.simulation.mode).",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel simulation trajectories.",
    )
    parser.add_argument(
        "--num-time-steps",
        type=int,
        default=None,
        metavar="N",
        help="Number of simulation time steps (after warm-up).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Global RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to write output files.",
    )

    return parser


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML config file and return it as a plain dict."""
    with path.open() as f:
        return yaml.safe_load(f)


def _coerce_value(raw: str) -> int | float | bool | str:
    """Best-effort type coercion of a string CLI value."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-notation key path."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_set_overrides(
    config: dict[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    """Apply ``--set KEY VALUE`` overrides (dot-notation) to a config dict.

    Values are cast to int / float / bool when possible; otherwise kept as str.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"--set argument must be in KEY=VALUE format, got: {override!r}"
            )
        key, _, raw_value = override.partition("=")
        _set_nested(config, key.strip(), _coerce_value(raw_value.strip()))
    return config


def apply_named_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply named ``--<param>`` overrides to the config dict.

    Each named argument maps to a dot-notation path in the config.
    """
    # Mapping: argparse dest → dot-notation config path
    # TODO: extend when more named parameters are added
    named_map: dict[str, str] = {
        "mode": "simulation.mode",
        "num_simulations": "simulation.num_simulations",
        "num_time_steps": "simulation.num_time_steps",
        "rng_seed": "simulation.rng_seed",
        "output_dir": "output.output_dir",
    }
    for dest, config_path in named_map.items():
        value = getattr(args, dest, None)
        if value is not None:
            _set_nested(config, config_path, value)
    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns an exit code (0 = success)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Load and merge configuration
    config_dict = load_config(args.config)
    config_dict = apply_set_overrides(config_dict, args.overrides)
    config_dict = apply_named_overrides(config_dict, args)

    simulator = RenewalSimulator.from_config_dict(config_dict)

    print(simulator.config.temporal.sim_start)


    # TODO: Build params_df (fixed + LHS-sampled parameters)
    # TODO: Build initial_infec_df
    # TODO: Load observations_sr if mode == "calibration"
    # TODO: Instantiate RenewalSimulator and call .run(...)
    # TODO: Persist SimulationOutput to output_dir
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
