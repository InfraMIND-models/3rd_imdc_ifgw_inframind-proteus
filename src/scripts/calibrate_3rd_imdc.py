"""
Calibration procedure of the outbreak dynamics model (renewal equation) of
Inframind Proteus.

This script is originally developed for the validation round of the
3rd Infodengue-Mosqlimate Dengue Challenge (3rd IMDC).

Usage
-----

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

import pydantic


class _BaseConfig:
    """Base class for nested configuration objects.

    Useful for representing data hierarchically (JSON/YAML-like) with
    structured fields.
    """

    def update_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Update the configuration from a dictionary.
        New fields are added, regardless of whether they exist in the current
        config or not.
        If a field matches another config object, the update is applied
        recursively to that config object.
        """
        for field_name, field_value in config_dict.items():
            is_cfg = False
            if hasattr(self, field_name):
                attr = getattr(self, field_name)
                if issubclass(attr.__class__, _BaseConfig):
                    attr.update_from_dict(field_value)
                    is_cfg = True

            if not is_cfg:
                setattr(self, field_name, field_value)

            if isinstance(field_value, _BaseConfig):
                field_value.update_from_dict(config_dict.get(field_name, {}))

    def preprocess(self, *args, **kwargs):
        """Note: Call super when overriding."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, _BaseConfig):
                field_value.preprocess(*args, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], preprocess=True) -> _BaseConfig:
        """Create a config object from a dictionary."""
        cfg = cls()
        cfg.update_from_dict(config_dict)
        if preprocess:
            cfg.preprocess()
        return cfg


@dataclass
class ProgramConfig(_BaseConfig):
    """Internal configuration data class for the calibration procedure script."""


    class Stage1Config(_BaseConfig):
        """"""
    stage1: Stage1Config = Stage1Config()

    class Stage2Config(_BaseConfig):
        """"""
    stage2: Stage2Config = Stage2Config()


class CommandLineArgs:
    """Command-line arguments for the calibration procedure script."""

    def get_argparser(self):
        parser = argparse.ArgumentParser()

        # parser.add_argument(
        #
        # )

        return parser


class ProgramData:
    """Internal payload data class for the calibration procedure script."""


def main(argv: list[str] | None = None) -> None:

    # # Test config
    # cfg = ProgramConfig(
    #     stage1=ProgramConfig.Stage1Config(
    #
    #     )
    # )

    cfg_dict = {
        "hello": "world",
        "stage1": {
            "param1": 123,
            "param2": "abc",
            "param3": {
                "subparam1": 0.5,
                "subparam2": [1, 2, 3],
            },
        }
    }

    cfg = ProgramConfig()
    cfg.update_from_dict(cfg_dict)

    cfg.preprocess()

    # cfg.stage1: ProgramConfig.Stage1Config
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
