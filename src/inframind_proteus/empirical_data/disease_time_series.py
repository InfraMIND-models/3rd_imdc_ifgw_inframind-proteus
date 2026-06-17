"""
Represent time series of epidemiological diseases data (number of cases, infections,
hospitalizations, etc.)
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DiseaseTimeSeriesVariables:
    """Variable names and metadata for one disease time series.

    Formalizes the variable contract for disease time series data, specifying
    the expected column names and metadata.

    Attributes
    ----------
    location_id_variable : str
        Name of the column containing location identifiers (default: "uf").
    time_variable : str
        Name of the column containing time/date information (default: "date").
    value_variable : str
        Name of the column containing the measured values (default: "value").
    disease : str
        Name of the disease being tracked (default: "dengue").
    """
    location_id_variable: str = "uf"
    time_variable: str = "date"
    value_variable: str = "value"
    disease: str = "dengue"


class DiseaseTimeSeriesCache:
    """Cache for loading and storing disease time series data by location.

    Loads CSV files containing disease case data and caches them in memory
    to avoid repeated file I/O operations.

    Parameters
    ----------
    disease_cases_dir : Path
        Directory containing the disease case CSV files.
    fname_fmt : str, optional
        Format string for constructing filenames. Should contain a
        `{location_id}` placeholder (default: "dengue_{location_id}.csv").
    variables : DiseaseTimeSeriesVariables | None, optional
        Variable configuration for the time series. If None, uses default
        DiseaseTimeSeriesVariables instance.

    Attributes
    ----------
    disease_cases_dir : Path
        Directory path for disease case files.
    fname_fmt : str
        Filename format string.
    variables : DiseaseTimeSeriesVariables
        Variable configuration instance.
    cache : dict
        Internal cache storing loaded time series data by location_id.
    """

    def __init__(
            self,
            disease_cases_dir: Path = Path("data/disease/dengue_cases_uf_weekly"),
            fname_fmt: str = "dengue_{location_id}.csv",
            variables: DiseaseTimeSeriesVariables | None = None,
    ):
        self.disease_cases_dir = Path(disease_cases_dir)
        self.fname_fmt = fname_fmt
        self.variables = variables or DiseaseTimeSeriesVariables()
        self.cache = {}

    def get_location(self, location_id: str) -> pd.Series:
        """Load and cache time series data for a specific location.

        Reads the CSV file for the given location if not already cached,
        parses dates if applicable, and returns the time series as a pandas Series.

        Parameters
        ----------
        location_id : str
            Identifier for the location (e.g., state code, region name).

        Returns
        -------
        pd.Series
            Time series indexed by the time variable, containing the values
            for the specified location.
        """
        if location_id not in self.cache:
            parse_dates = list()
            if self.variables.time_variable in ["date"]:
                parse_dates.append(self.variables.time_variable)

            cases_df = pd.read_csv(
                self.disease_cases_dir / self.fname_fmt.format(location_id=location_id),
                parse_dates=parse_dates,
            )
            cases_sr = cases_df.set_index(self.variables.time_variable)[self.variables.value_variable]
            self.cache[location_id] = cases_sr

        return self.cache[location_id]
