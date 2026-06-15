import os
import gc
import importlib
from copy import deepcopy
from pathlib import Path


import epiweeks
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from inframind_proteus.outbreak_dynamics import RenewalSimulator, build_calibration_params_df, build_initial_infec_df
from inframind_proteus.outbreak_dynamics.utils import (
    load_yaml_dict, apply_include_exclude_logic, map_parallel_or_sequential_chunks, save_yaml_dict,
    map_parallel_or_sequential
)


def main():
    # Parameters of the multi-location code
    # ================

    base_config_path = Path("configs/prototype_run_dynamic_model.yaml")
    base_config_dict = load_yaml_dict(Path("configs/prototype_run_dynamic_model.yaml"))
    # cases_cache = DiseaseTimeSeriesCache(Path("data/disease/dengue_cases_uf_weekly"))

    zero_date_epiweek = 41
    sim_start_epiweek = 26  # Using zero_date minus at least 2x the maximum generation time in weeks
    calibration_start_epiweek = 41
    calibration_end_epiweek = 40  # Of the next year

    # use_location_ids = ["SP", "SE", "PI"]
    use_location_ids = []  # Runs all!
    exclude_location_ids = [] # ["CE"]  # Just to test.
    use_years = list(range(2016, 2025))


    # ===========================

    # ==========

    # Load UF table to lookup locations
    uf_table_df = pd.read_csv(Path("data/demographic/uf_table.csv"))

    # Combine all years and locations to be run
    location_ids = apply_include_exclude_logic(
        uf_table_df["uf"],
        include_list=use_location_ids,
        exclude_list=exclude_location_ids,
    )
    years = list(use_years)
    location_year_index = pd.MultiIndex.from_product(
        [location_ids, years],
        names=["uf", "season"]
    )

    def run_simulation_for_location_year(location_year_tuple):
        location_id, year = location_year_tuple
        print(f"Iterating for {location_id} in {year}...")

        # Load again observations, since this function will run in parallel... not worth caching
        obserations_sr = DiseaseTimeSeriesCache(Path("data/disease/dengue_cases_uf_weekly")).get(location_id)

        # Create and modify the simulator and config object
        config_dict = deepcopy(base_config_dict)
        config_dict["location"]["location_id"] = location_id
        config_dict["simulation"]["population_size"] = uf_table_df.set_index("uf").loc[
            location_id, f"population_{year}"].item()
        _temporal = config_dict["temporal"]
        _temporal["zero_date"] = epiweeks.Week(year, zero_date_epiweek).startdate().isoformat()
        _temporal["sim_start"] = epiweeks.Week(year, sim_start_epiweek).startdate().isoformat()
        _temporal["calibration_start"] = epiweeks.Week(year, calibration_start_epiweek).startdate().isoformat()
        _temporal["calibration_end"] = epiweeks.Week(year + 1, calibration_end_epiweek).startdate().isoformat()

        simulator = RenewalSimulator.from_config_dict(config_dict)
        config = simulator.config

        # --- Build auxiliary data frames for simulations
        params_df = build_calibration_params_df(config.num_simulations, config.sampling)
        config.num_simulations = num_simulations = params_df.shape[
            0]  # Update in case sampling method changes number of simulations
        initial_infec_df = build_initial_infec_df(
            config.num_simulations, simulator._gt_max_steps, config.temporal.step_dt,
            config.initial_infections
        )

        # Run
        # ======
        results = simulator.run(
            params_df=params_df,
            initial_infec_df=initial_infec_df,
            observations_sr=obserations_sr,
        )

        # [PROTOTYPE] Select best simulations via WIS
        wis_sr = results.scoring.summary["wis"]
        # -()- By fraction of number of simulations
        _nsim = wis_sr.shape[0]
        _frac = 0.001
        _n = np.ceil(_frac * _nsim).astype(int)

        selected_wis_sr = wis_sr.nsmallest(_n)

        # [PROTOTYPE]: Export results
        # =====
        _root = Path(".")
        out_dir = _root / Path(config_dict["output"]["main_dir"]) / "calibration_results" / f"{location_id}_{year}"
        out_dir.mkdir(exist_ok=True, parents=True)

        save_yaml_dict(config_dict, out_dir / "config.yaml")

        params_df.to_csv(out_dir / "params.csv.gz")  # Also kinda heavy!

        # results.scoring.summary.to_parquet(out_dir / "scoring.parquet")
        results.scoring.summary.to_csv(out_dir / "scoring.csv.gz")

        # results.case_beam_df.to_csv(out_dir / "case_beam_df.csv")  # TOOOOOOOOO heavy!

        # Export only selected trajectories (case beams) to save space
        results.case_beam_df.reset_index().set_index("i_simulation").loc[selected_wis_sr.index].to_csv(
            out_dir / "case_beam_selected_df.csv.gz")
        print(f"Done: {out_dir}")


    def _task(location_year_tuple):
        try:
            run_simulation_for_location_year(location_year_tuple)
        except Exception as e:
            print(f"Error for {location_year_tuple}: {e}")
            raise e
        finally:
            gc.collect()  # Try to free memory after each simulation

    # -------
    map_parallel_or_sequential_chunks(
    # map_parallel_or_sequential(
        _task,
        location_year_index,
        ncpus=5, chunksize=1
    )
    # run_simulation_for_location_year(("SP", 2023))
    print("Done all simulations!")



class DiseaseTimeSeriesCache:

    def __init__(
            self,
            disease_cases_dir: Path,
            fname_fmt: str = "dengue_{uf}.csv"
    ):
        self.disease_cases_dir = Path(disease_cases_dir)
        self.fname_fmt = fname_fmt
        self.cache = {}

    def get(self, uf: str) -> pd.Series:
        if uf not in self.cache:
            cases_df = pd.read_csv(
                self.disease_cases_dir / self.fname_fmt.format(uf=uf),
                parse_dates=["date"],
            )
            cases_sr = cases_df.set_index("date")["value"]
            self.cache[uf] = cases_sr
        return self.cache[uf]



if __name__ == "__main__":
    main()
