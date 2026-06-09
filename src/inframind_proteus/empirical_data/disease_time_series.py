"""
Represent time series of epidemiological diseases data (number of cases, infections,
hospitalizations, etc.)
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#TODO: Needs to be implemented

@dataclass
class DiseaseTimeSeriesVariables:
    """Variable names and metadata for one disease time series.

    Formalizes the variable contract.
    """
    location_id_variable: str = "geocode"
    time_variable: str = "date"
    value_variable: str = "value"
    disease: str = "dengue"


# class DiseaseTimeSeries:
#     """"""
#
#
#     _df: pd.DataFrame
#
#     @classmethod
#     def from_csv_file(cls, fpath: Path):
#         """"""
#
#         fpath = Path(fpath)
#         if not fpath.exists():
#             raise FileNotFoundError(f"Disease time series file not found: {fpath}")
#
#         df = pd.read_csv(fpath)

