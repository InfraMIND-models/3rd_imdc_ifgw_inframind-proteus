"""
Represent time series of epidemiological diseases data (number of cases, infections,
hospitalizations, etc.)
"""
from pathlib import Path

import pandas as pd

#TODO: Needs to be implemented

class DiseaseTimeSeries:
    """"""
    location_id_variable: str
    time_variable: str
    value_variable: str
    disease: str

    _df: pd.DataFrame

    @classmethod
    def from_csv_file(cls, fpath: Path):
        """"""

