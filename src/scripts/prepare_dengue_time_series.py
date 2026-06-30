"""Preprocess the dengue time series data into InfraMIND-internal format.

Starts from dengue time series as provided by the IMDC organization.
"""
import os
from pathlib import Path

import pandas as pd

from inframind_proteus.empirical_data import DiseaseTimeSeriesVariables

class Config:
    def __init__(self):
        # Locate project root directory
        self.root_dir = Path(__file__).resolve().parent.parent.parent

        # self.imdc_regional_fpath = Path("data/data_imdc_2026/map_regional_health.csv")
        self.imdc_population_fpath = Path("data/data_imdc_2026/datasus_population_2001_2025.csv.gz")
        self.imdc_dengue_fpath = Path("data/data_imdc_2026/dengue.csv.gz")

        self.uf_dengue_dirpath = Path("data/disease/dengue_cases_uf_weekly")
        self.uf_dengue_fname_fmt = "dengue_{uf}.csv"

        self.do_export = True

def main():
    config = Config()
    os.chdir(config.root_dir)
    print("Working directory: ", os.getcwd())

    dvars = DiseaseTimeSeriesVariables()
    # imdc_regional_df = pd.read_csv(config.imdc_regional_fpath)
    imdc_population_df = pd.read_csv(config.imdc_population_fpath)
    imdc_dengue_df = pd.read_csv(config.imdc_dengue_fpath, parse_dates=["date"])

    uf_dengue_df = aggregate_dengue_to_uf(
        imdc_dengue_df, imdc_population_df, dvars
    )

    export_dengue_uf_weekly(
        uf_dengue_df, config.uf_dengue_dirpath, config.uf_dengue_fname_fmt, dvars
    )


def aggregate_dengue_to_uf(
        imdc_dengue_df: pd.DataFrame,
        imdc_population_df: pd.DataFrame,
        dvars: DiseaseTimeSeriesVariables,
):
    """"""
    df = imdc_dengue_df.copy()

    # --- Add year and population data
    df["year"] = df["date"].dt.year
    df = df.merge(imdc_population_df, on=["geocode", "year"])

    # --- Aggregate and calculate per-populaiton incidence
    df = (
        df.groupby(["uf", "date"])
        .agg({"casos": "sum", "population": "sum"})
        .reset_index()
    )
    # state_sr = df.groupby(["uf", "date"])["casos"].sum()

    df["case_inc_100k"] = df["casos"] / df["population"] * 1E5

    # --- Standardize variable names
    df = df.rename(columns={
        "casos": dvars.value_variable,
        # "date": dvars.time_variable,
    })

    return df


def export_dengue_uf_weekly(
        uf_dengue_df: pd.DataFrame,
        uf_dengue_dirpath: Path,
        uf_dengue_fname_fmt: str,
        dvars: DiseaseTimeSeriesVariables,
):
    """"""

    for uf, uf_df in uf_dengue_df.groupby("uf"):
        uf_fpath = uf_dengue_dirpath / uf_dengue_fname_fmt.format(uf=uf)
        uf_fpath.parent.mkdir(parents=True, exist_ok=True)
        uf_df[[dvars.time_variable, dvars.value_variable]].to_csv(uf_fpath, index=False)
        print(f"Exported: {uf_fpath}")


if __name__ == "__main__":
    main()
