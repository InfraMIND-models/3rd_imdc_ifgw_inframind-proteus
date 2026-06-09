"""
Represent time series of epidemiological diseases data (number of cases, infections,
hospitalizations, etc.)
"""
from dataclasses import dataclass


@dataclass
class DiseaseTimeSeriesVariables:
    """Variable names and metadata for one disease time series.

    Formalizes the variable contract.
    """
    location_id_variable: str = "uf"
    time_variable: str = "date"
    value_variable: str = "value"
    disease: str = "dengue"
