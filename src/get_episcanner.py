"""Script to download data from the Episcanner dataset using the mosqlient interface.
Source: https://info.dengue.mat.br/epi-scanner/

Requires a Mosqlimate API key as `MOSQLIMATE_API_KEY` environment variable.
Register and get a free key following this guide: https://api.mosqlimate.org/docs/overview/
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
import mosqlient
import pandas as pd


def main():
    year_range = [2011, 2027]  # Range of years to download (start inclusive, end exclusive)
    disease = "dengue"
    sleep_seconds = 1  # Sleep time between API calls to avoid rate limits and be polite to the server.

    # ---
    load_dotenv()
    api_key = os.environ.get("MOSQLIMATE_API_KEY")

    # ---
    uf_table_df = pd.read_csv("data/uf_table.csv")

    # ---

    # keys_list = list()
    df_list = list()
    for year in range(year_range[0], year_range[1]):
    # for year in range(2015, 2017):  # TEST
        for uf in uf_table_df["uf"]:
        # for uf in uf_table_df["uf"].iloc[0:2]:  # TEST
            print(f"Downloading data for year {year} and UF {uf}...")

            df = mosqlient.get_episcanner(
                api_key=api_key,
                disease=disease,
                uf=uf,
                year=year,
            )
            df["uf"] = uf

            # keys_list.append(uf)
            df_list.append(df)


            time.sleep(sleep_seconds)  # Sleep to make polite API calls

    # Concatenate all dataframes into one.
    full_df = pd.concat(df_list, ignore_index=True)

    # Rearange and save
    other_cols = [col for col in full_df.columns if col not in ["uf", "year"]]
    full_df = full_df[["uf", "year", *other_cols]]

    fpath = Path(f"data/episcanner/episcanner_dengue_{year_range[0]}_{year_range[1]-1}.csv")
    fpath.parent.mkdir(exist_ok=True, parents=True)
    full_df.to_csv(fpath, index=False)

    # print("WATCHPOINT")

if __name__ == "__main__":
    main()
