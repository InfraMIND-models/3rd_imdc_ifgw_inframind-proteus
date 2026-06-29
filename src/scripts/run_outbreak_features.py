#!/usr/bin/env python
"""CLI entry point for the inframind-proteus outbreak-features component.

Produces stochastic-sample predictions of the macroscopic outbreak features per (UF, year)
and writes them in the export-contract schema.

Usage
-----
    uv run python src/scripts/run_outbreak_features.py
    uv run python src/scripts/run_outbreak_features.py --n-samples 500 --years 2022 2023 2024 2025
    uv run run-outbreak-features --output-dir predictions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inframind_proteus.outbreak_features import config
from inframind_proteus.outbreak_features.data import DataRepository
from inframind_proteus.outbreak_features.export import DEFAULT_YEARS, build_features_export
from inframind_proteus.outbreak_features.contract import write_contract


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_outbreak_features",
        description="Run the inframind-proteus outbreak-features component.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--n-samples", type=int, default=500, metavar="N",
                   help="Number of stochastic samples per (UF, year, feature). Default 500.")
    p.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS), metavar="Y",
                   help="Validation start years to predict. Default 2022 2023 2024 2025.")
    p.add_argument("--level", default="state", choices=list(config.SPATIAL_LEVELS),
                   help="Spatial level. The contract is at state (UF) level. Default state.")
    p.add_argument("--seed", type=int, default=0, metavar="SEED", help="Global RNG seed.")
    p.add_argument("--output-dir", type=Path, default=None, metavar="DIR",
                   help="Directory for the contract CSVs. Default <repo>/predictions.")
    p.add_argument("--samples-out", type=Path, default=None, metavar="PATH",
                   help="Optional path to also save the intermediate rate/week samples.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    samples = build_features_export(years=tuple(args.years), n_samples=args.n_samples,
                                    level=args.level, seed=args.seed)

    if args.samples_out is not None:
        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(args.samples_out, index=False)
        print(f"saved intermediate rate/week samples -> {args.samples_out}")

    repo = DataRepository(args.level)
    paths = write_contract(samples, repo, out_dir=args.output_dir)
    print(f"\nwrote {len(paths)} contract files:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
