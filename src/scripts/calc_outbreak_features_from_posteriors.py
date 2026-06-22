"""Given a set of parameters from the outbreak dynamic model, calculate
distributions of outbreak features for each parameter set.

Mainly designed to calculate outbreak features after `calibrate-3rd-imdc`
has been run.

"""

import argparse
import re
import sys
from pathlib import Path
from typing import Union, Any, Literal

import numpy as np
import pandas as pd

from inframind_proteus import BaseConfig
from inframind_proteus.outbreak_dynamics import SimulationConfig
from inframind_proteus.outbreak_dynamics.outbreak_features import outbreak_features_from_trajectories, \
    outbreak_feature_stats
# from inframind_proteus.empirical_data import DiseaseTimeSeriesCache
from inframind_proteus.outbreak_dynamics.utils import load_yaml_dict, parse_set_arguments_with_yaml, year_week_to_date


class ProgramConfig(BaseConfig):
    """"""
    run_datetime: pd.Timestamp
    # config_fpath: Union[Path, None] = None # Path("configs/calibrate_3rd_imdc_default.yaml")
    config_fpath: Path
    #  ^  Specifying a config is optional. If not specified, use only defaults
    #     and command line arguments.
    output_dir: Path
    #  ^ Path to the location-year specific output directory.
    #    Calibration stage 3 outputs are expected to be stored there.
    location_id: Any
    year: int

    out_dir_fmt: str = "{location_id}_{year}"
    # # ^ In case out_dir is not provided, construct from this format string.

    param_samples_fname: str = "stage3_posterior_samples.csv.gz"
    mean_cases_fname: str = "stage3_mean_cases.csv.gz"
    sim_config_fname: str = "stage3_sim_config.yaml"

    outbreak_feature_names : list[str] = [
        "case_attack_rate",
        "peak_week",
        "peak_amplitude",
    ]
    #  ^ Define which features should be calculated.

    # --- Temporal config, epiweek-based (year-agnostic)
    calculation_start_epiweek: int = 41
    calculation_end_epiweek: int = 40  # Of the next year
    end_in_next_year: bool = True  # Whether the calculation end epiweek is in the next year (e.g. 40 of the next year)

    num_stochastic_trajectories = 100

    export_feature_samples: bool = False
    export_feature_stats: bool = True
    # export_

    # disease_cases_dir: Path = Path("data/disease/dengue_cases_uf_weekly")

    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)

        self.run_datetime = pd.Timestamp.now(tz="UTC")

        # If year was not provided, try to assert it from the output_dir name
        # using the location-year format.
        if not hasattr(self, "year") or self.year is None:
            if hasattr(self, "output_dir") and self.output_dir is not None:
                # Try to extract year from output_dir name using the location-year format.
                fmt = self.out_dir_fmt
                # Create a regex pattern from the format string.
                pattern = fmt.replace("{location_id}", r"(?P<location_id>.+)")
                pattern = pattern.replace("{year}", r"(?P<year>\d{4})")
                match = re.match(pattern, self.output_dir.name)
                if match:
                    self.year = int(match.group("year"))
                    self.location_id = match.group("location_id")
                    print(
                        f"Inferred year={self.year} and "
                        f"location_id={self.location_id} from output_dir name: "
                        f"{self.output_dir}"
                    )
                else:
                    raise ValueError(
                        f"Could not extract year and location_id from output_dir"
                        f" name '{self.output_dir.name}' "
                        f"using the format '{fmt}'. Please provide the year and "
                        f"location_id explicitly via command line or config."
                    )
            else:
                raise ValueError(
                    "Year was not provided and could not be inferred from "
                    "output_dir. "
                    "Please provide the year explicitly or ensure output_dir "
                    "is set correctly."
                )



class ProgramData:
    """Internal payload data class for the calibration procedure script."""

    sim_cfg: SimulationConfig
    param_samples_df: pd.DataFrame
    mean_cases_df: pd.DataFrame
    # observations_sr: pd.Series  # Indexed by date, values are case counts

    augm_param_samples_df: pd.DataFrame
    augm_mean_cases_df: pd.DataFrame
    # ^ Augmented so each row represents a trajectory to calculate outbreak features from

    cases_df: pd.DataFrame
    # ^ Stochastic trajectories of disease cases to calculate outbreak features.
    #    Signature: df.loc[(*trajetory_id_vars*), time]

    feature_samples_df: pd.DataFrame
    # ^ Outbreak features from each individual trajectory (full data).
    #   Signature: df.loc[(*trajetory_id_vars*), feature_name]
    feature_stats_df: pd.DataFrame
    # ^ Summary statistics of outbreak features.
    #   Signature: df.loc[(*groupby_vars*), (feature_name, stat_name)] = stat_value


def parse_args_get_dict(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse command-line arguments and return them as a dictionary."""
    parser = argparse.ArgumentParser()


    # --- Config file path
    parser.add_argument(
        "--config-fpath", "--cfg", "-c",
        # default=ProgramConfig.config_fpath,
        type=Path,
        help="Path to the calibration YAML configuration file.",
    )

    parser.add_argument(
        "--output-dir", "--out", "-o",
        # default=ProgramConfig.output_dir,
        required=True,
        type=Path,
        help="Path to the location-year specific output directory.",
    )

    parser.add_argument(
        "--location-id", "-l",
        type=str,
        help="Location ID (by default, UF code) for the location to process.",
    )

    parser.add_argument(
        "--year", "-y",
        type=int,
        help="Year of the location-year to process.",
    )

    # --- Generic nested `--set` argument.
    parser.add_argument(
        "--set",
        nargs=2,
        action="append",
        default=list(),
        metavar=("KEY", "VALUE"),
        help="Set a configuration parameter using dot notation (e.g. "
             "--set stage1.num_simulations 4096)."
             " Can be used multiple times.",
    )


    # ======

    args = parser.parse_args(argv)
    # Retain only informed arguments to avoid overriding config.
    args_dict = {k: v for k, v in args.__dict__.items() if v is not None}

    # Proceess --set arguments to override config values.
    set_args = args_dict.pop("set")
    overrides = parse_set_arguments_with_yaml(set_args)
    args_dict.update(overrides)

    return args_dict


def main(argv: Union[list[str], None] = None) -> None:
    pass

    # --- Program initialization sequence
    args_dict = parse_args_get_dict(argv)
    cfg = ProgramConfig()
    if "config_fpath" in args_dict:
        config_dict = load_yaml_dict(args_dict["config_fpath"])
        cfg.update_from_dict(config_dict)
    cfg.update_from_dict(args_dict)
    cfg.preprocess()
    data = ProgramData()

    # out_dir = main_out_dir / f"{location_id}_{year}"

    # Load main data
    # ====================
    if not cfg.output_dir.exists():
        raise FileNotFoundError(
            f"Output directory {cfg.output_dir} does not exist. "
            f"Run the calibration procedure for this location-year first."
        )

    sim_config_dict = load_yaml_dict(
        cfg.output_dir / cfg.sim_config_fname, safe=False
    )
    data.sim_cfg = SimulationConfig.from_dict(sim_config_dict)

    data.param_samples_df = pd.read_csv(
        cfg.output_dir / cfg.param_samples_fname,
        index_col=0,
    )

    data.mean_cases_df = pd.read_csv(
        cfg.output_dir / cfg.mean_cases_fname,
        index_col=0,
    )
    data.mean_cases_df.columns = pd.to_datetime(data.mean_cases_df.columns)

    _validate_main_data(data)
    # observations_sr = DiseaseTimeSeriesCache(disease_cases_dir).get_location(location_id)

    # Sample trajectories to calculate features from
    # ===================
    data.augm_param_samples_df, data.augm_mean_cases_df = (
        _trajectories_from_each_sample(cfg, data)
    )

    # Define calculation window
    # ===================
    # -()-
    window_start = year_week_to_date(
        cfg.year, cfg.calculation_start_epiweek
    )
    window_end = year_week_to_date(
        cfg.year + cfg.end_in_next_year,
        cfg.calculation_end_epiweek
    )
    zero_date = pd.Timestamp(data.sim_cfg.temporal.zero_date)


    # Construct trajectories and calculate outbreak features
    # ===============
    data.cases_df = _apply_stochastic_observation_model(
        cfg, data
    )

    data.feature_samples_df = outbreak_features_from_trajectories(
        cases_df=data.cases_df,
        zero_date=zero_date,
        window_start=window_start,
        window_end=window_end,
        feature_names=cfg.outbreak_feature_names,
    )

    data.feature_stats_df = outbreak_feature_stats(
        data.feature_samples_df
    )

    # ==========
    _export_data(cfg, data)

    return


def _validate_main_data(
        data: ProgramData
):
    """"""
    sim_cfg = data.sim_cfg
    param_samples_df = data.param_samples_df
    mean_cases_df = data.mean_cases_df

    # --- Validate sim_cfg
    if sim_cfg.observation_model.model != "negative_binomial":
        print(
            "WARNING: Currently only supports negative binomial observation model, "
            "but got '{sim_cfg.observation_model.model}'. Will perform calculations"
            " but they may not match the model used to generate the data."
        )

    # --- Validate param_samples_df
    _required = [
        "notif_nb_overdispersion",
        "posterior_weight",
    ]
    _missing = [name for name in _required if name not in param_samples_df.columns]
    if _missing:
        raise ValueError(
            f"Required columns {', '.join(_missing)} not found in parameter samples dataframe."
        )

    # --- Validate mean_cases_df
    if mean_cases_df.index.name != "i_simulation":
        raise ValueError(
            f"Currently only supports `mean_cases_df` with an 'i_simulation' index, "
            f"but got '{mean_cases_df.index.name}'."
        )


def _trajectories_from_each_sample(
        cfg: ProgramConfig,
        data: ProgramData,
):
    """Get trajectories to calculate outbreak features from.
    Each parameter set of the posterior gets `n_stochastic` independent
    stochastic trajectories.
    """

    mean_cases_df = data.mean_cases_df
    param_samples_df = data.param_samples_df
    i_sim_var = "i_simulation"  # Must be the simulation id index in all data frames
    n_stochastic = cfg.num_stochastic_trajectories

    # Expand index to both simulation id and stochastic trajectory id.
    augm_index = pd.MultiIndex.from_product(
        [mean_cases_df.index, range(n_stochastic)], names=[i_sim_var, "i_stochastic"]
    )
    #   ^ ^ New index signature: (i_simulations, i_stochastic)
    i_sim_index = augm_index.get_level_values(i_sim_var)

    # Augment the time series of new cases
    augm_mean_cases_df = mean_cases_df.reindex(i_sim_index)
    augm_mean_cases_df.index = augm_index

    # Augment the posterior samples (parameter sets)
    augm_param_samples_df = param_samples_df.reindex(i_sim_index)
    augm_param_samples_df.index = augm_index

    return augm_param_samples_df, augm_mean_cases_df


def _apply_stochastic_observation_model(
        cfg: ProgramConfig,
        data: ProgramData,
):
    augm_mean_cases_df = data.augm_mean_cases_df
    augm_param_samples_df = data.augm_param_samples_df

    # Apply the 2D reshaping to sample trajectories with vectorized numpy
    # =======================
    # Standard shape: (i_augmented_sample, i_time)
    _expectancy = augm_mean_cases_df.values
    _overdisp = augm_param_samples_df["notif_nb_overdispersion"].to_numpy()[:, np.newaxis]
    p = _overdisp / (_overdisp + _expectancy)

    # Sample with negative binomial (observation model)
    # ============================
    rng = np.random.default_rng(seed=42)
    cases_vec: np.ndarray = rng.negative_binomial(
        n=_overdisp, p=p, size=_expectancy.shape
    )
    cases_df = pd.DataFrame(
        cases_vec,
        index=augm_mean_cases_df.index,
        columns=augm_mean_cases_df.columns
    )

    return cases_df


def _export_data(
        cfg: ProgramConfig,
        data: ProgramData,
):
    if cfg.export_feature_samples and hasattr(data, "feature_samples_df"):
        _fpath = cfg.output_dir / "outbreak_feature_samples.csv.gz"
        data.feature_samples_df.to_csv(
            _fpath,
            index=True,
            compression={
                "method": "gzip",
                "compresslevel": 9,
            }
        )
        print(f"Exported: {_fpath}")


    if cfg.export_feature_stats and hasattr(data, "feature_stats_df"):
        # _fpath = cfg.output_dir / "outbreak_feature_stats.csv.gz"
        _fpath = cfg.output_dir / "outbreak_feature_stats.csv"
        data.feature_stats_df.to_csv(
            _fpath,
            index=True,
            compression={
                "method": "gzip",
                "compresslevel": 9,
            }
        )
        print(f"Exported: {_fpath}")

if __name__ == '__main__':
    main(sys.argv[1:])
