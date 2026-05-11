"""Prepare an auxiliary table file with metadata from each Federative Unit (UF),
starting from IMDC data.
"""
from pathlib import Path

import pandas as pd


def main():
    # --- Parameters
    year_range = [2010, 2026]

    # --- Load IMDC data
    imdc_regional_df = pd.read_csv("data/data_imdc_2026/map_regional_health.csv")
    imdc_population_df = pd.read_csv("data/data_imdc_2026/datasus_population_2001_2025.csv.gz")

    # One UF-level row with UF and macroregion metadata plus municipality count.
    uf_df = (
        imdc_regional_df.groupby(["uf", "uf_name", "uf_code"], as_index=False)
        .agg(
            num_municipalities=("geocode", "nunique"),
            macroregion_code=("macroregion_code", "first"),
            macroregion_name=("macroregion_name", "first"),
        )
    )

    # Map municipality geocode to UF.
    geocode_to_uf = imdc_regional_df[["geocode", "uf"]].drop_duplicates()

    # Filter requested years (excluding range end), aggregate municipality populations to UF-year,
    # and pivot to population_yyyy columns.
    years = list(range(year_range[0], year_range[1]))
    uf_population_wide_df = (
        imdc_population_df.loc[imdc_population_df["year"].isin(years)]
        .merge(geocode_to_uf, on="geocode", how="left")
        .dropna(subset=["uf"])
        .groupby(["uf", "year"], as_index=False)["population"]
        .sum()
        .pivot(index="uf", columns="year", values="population")
        .rename(columns=lambda y: f"population_{int(y)}")
        .reset_index()
    )

    # Keep column order with yearly population columns at the end.
    population_columns = [f"population_{y}" for y in years]
    uf_table_df = uf_df.merge(uf_population_wide_df, on="uf", how="left")
    for col in population_columns:
        if col not in uf_table_df.columns:
            uf_table_df[col] = pd.NA

    uf_table_df = uf_table_df[
        [
            "uf",
            "uf_name",
            "uf_code",
            "num_municipalities",
            "macroregion_code",
            "macroregion_name",
            *population_columns,
        ]
    ]

    # --- Export
    print(uf_table_df)
    fpath = Path("data/uf_table.csv")
    fpath.parent.mkdir(parents=True, exist_ok=True)
    uf_table_df.to_csv(fpath, index=False)
    print(f"Data exported to {fpath}")


if __name__ == "__main__":
    main()
