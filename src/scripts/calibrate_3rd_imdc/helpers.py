from pathlib import Path

import pandas as pd

from inframind_proteus.outbreak_dynamics.utils import year_week_to_date
from .program_config import ProgramConfig


# Internal auxiliary functions
# =========================================
def _set_config_dict_common(
        cfg: ProgramConfig,
        sim_config_dict: dict,
        location_id,
        year,
        uf_table_df: pd.DataFrame,
        num_simulations: int | None = None,
        scoring_metrics: list[str] | None = None
):
    """Override simulation configuration dictionary fields with parameters
    that are common to all calibration stages.
    """
    _d = sim_config_dict

    def _todate(y, w):
        return year_week_to_date(y, w).date().isoformat()

    # --- Set location-specific config fields
    _d["location"]["location_id"] = location_id
    _d["location"]["population_size"] = (
        uf_table_df
        .set_index("uf")
        .loc[location_id, f"population_{year}"]
        .item()
    )

    # --- Set the time fields (simulation start date, calibration window, etc)
    _temporal = _d["temporal"]
    _temporal["zero_date"] = _todate(year, cfg.zero_date_epiweek)
    _temporal["sim_start"] = _todate(year, cfg.sim_start_epiweek)
    _temporal["calibration_start"] = _todate(year, cfg.calibration_start_epiweek)
    _temporal["calibration_end"] = _todate(year + 1, cfg.calibration_end_epiweek)

    if num_simulations is not None:
        _d["simulation"]["num_simulations"] = num_simulations

    if scoring_metrics is not None:
        _d["scoring"]["metrics"] = scoring_metrics


def prepare_output_subdirs(
        location_id, year,
        output_dir: Path,
        location_year_subdir_fmt: str,
        mkdirs=True,
):
    """Make output subdirectories names for a given location and year and
    return their paths. Optionally create the directories if they don't
    exist (default: True).
    """
    location_year_subdir = (
        output_dir
        / location_year_subdir_fmt.format(
            location_id=location_id, year=year
        )
    )

    data_out_dir = location_year_subdir
    plots_out_dir = location_year_subdir / "plots"

    if mkdirs:
        plots_out_dir.mkdir(exist_ok=True, parents=True)
        data_out_dir.mkdir(exist_ok=True, parents=True)

    return data_out_dir, plots_out_dir
