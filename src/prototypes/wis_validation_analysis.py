"""Prototype: score validation-round projections with WIS (+ secondary metrics),
comparing WITH vs. WITHOUT the outbreak-feature Bayesian update, and plot the
outbreak-feature predictive distributions themselves.

Standalone prototype: does not modify the production pipeline. Reads
already-computed predictions from two sources:

- "with update" (official validation round submission):
  outputs/validation_round_projections/projections/<UF>_<YEAR>/imdc_submission.csv

- "without update" (ablation, `outbreak_feature_names: []`):
  outputs/sensitivity/no_update_from_outbreak_features/projections/<UF>_<YEAR>/imdc_submission.csv
  generated via
    configs/alternative/process_data_for_projections_no-update.yaml
    configs/alternative/project_3rd_imdc_no-update.yaml

Plus raw inputs used for context/plots:
- predictions/{case_attack_rate,peak_amplitude,peak_week}.csv (outbreak-feature
  predictive samples that fed the Bayesian update)
- data/disease/dengue_cases_uf_weekly/dengue_<UF>.csv (observed weekly cases)
"""
from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from inframind_proteus.base_config import BaseConfig
from inframind_proteus.empirical_data import DiseaseTimeSeriesCache
from inframind_proteus.outbreak_dynamics.scoring import wis_score_vectorized, coverages_vectorized
from inframind_proteus.outbreak_dynamics.outbreak_features import (
    OutbreakFeaturePredictionsCache, outbreak_features_from_trajectories
)
from inframind_proteus.outbreak_dynamics.utils import year_week_to_date

# Project identity colors, borrowed from `revise_and_submit_projections.ipynb`.
PLOT_COLORS = dict(
    obs="#262626ff",
    median="#49786cff",
    pi95="#b6c6a0ff",
    pi50="#68aa80ff",
)


def main():
    cfg = ProgramConfig()
    cfg.preprocess()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    location_year_ids = get_location_year_ids(cfg)
    print(f"Found {len(location_year_ids)} location-season combos with predictions on both sides.")

    # --- WIS (+ secondary metrics) scoring, with vs. without the update
    scores_df = run_scoring(cfg, location_year_ids)
    scores_df.to_csv(cfg.output_dir / "wis_scores_long.csv", index=False)
    print(f"Exported per location-season scores: {cfg.output_dir / 'wis_scores_long.csv'}")

    summary_tables = build_summary_tables(scores_df)
    for name, df in summary_tables.items():
        fpath = cfg.output_dir / f"wis_summary_{name}.csv"
        df.to_csv(fpath)
        print(f"Exported summary table '{name}': {fpath}")

    plot_wis_comparison(scores_df, cfg.output_dir / "wis_comparison.pdf")
    print(f"Exported WIS comparison plots: {cfg.output_dir / 'wis_comparison.pdf'}")

    # --- Predictive time series by location (one PDF per scenario, one page per location)
    scenario_dirs = {"with_update": cfg.with_update_dir, "no_update": cfg.no_update_dir}
    for scenario, base_dir in scenario_dirs.items():
        predictions_long_df = load_predictions_long(cfg, base_dir, location_year_ids)
        out_fpath = cfg.output_dir / f"predictive_timeseries_{scenario}.pdf"
        plot_predictive_timeseries_by_location(cfg, scores_df, scenario, predictions_long_df, out_fpath)
        print(f"Exported predictive time series ({scenario}): {out_fpath}")

    # --- Outbreak-feature predictive distributions (all location-seasons)
    plot_outbreak_feature_distributions(
        cfg, location_year_ids, cfg.output_dir / "outbreak_feature_distributions.pdf"
    )
    print(
        "Exported outbreak-feature distribution plots: "
        f"{cfg.output_dir / 'outbreak_feature_distributions.pdf'}"
    )


class ProgramConfig(BaseConfig):
    """Configuration for the WIS validation-round analysis prototype."""

    # --- Prediction sources (submission-style CSVs, one per UF-season)
    with_update_dir: Path = Path("outputs/validation_round_projections/projections")
    no_update_dir: Path = Path("outputs/sensitivity/no_update_from_outbreak_features/projections")
    location_year_subdir_fmt: str = "{location_id}_{year}"
    submission_fname: str = "imdc_submission.csv"

    # --- Observed data
    disease_cases_dir: Path = Path("data/disease/dengue_cases_uf_weekly")

    # --- Outbreak-feature predictive samples (for the distribution plots)
    outbreak_features_predictions_dir: Path = Path("predictions")
    outbreak_feature_names: list[str] = ["case_attack_rate", "peak_week", "peak_amplitude"]

    # Epiweek references to align predicted `peak_week` samples with the dynamic
    # model's own zero-date, exactly mirroring the adjustment applied in
    # `scripts.process_data_for_projections._calc_likelihood_of_outbreak_features`.
    dyn_zero_date_epiweek: int = 41
    feat_peak_ref_epiweek: int = 40

    # --- Scope. Empty lists mean "use all available".
    use_location_ids: list[str] = []
    use_projection_years: list[int] = [2022, 2023, 2024, 2025]

    # --- Submission file format (mirrors scripts.project_3rd_imdc.ProgramConfig)
    imdc_median_prediction_colname: str = "pred"
    imdc_required_predictive_intervals: list[float] = [0.50, 0.80, 0.90, 0.95]
    imdc_pi_colname_fmt_dict: dict = {
        "lower": "lower_{level100:.0f}",
        "upper": "upper_{level100:.0f}",
    }
    # Submission window reference (only used to detect partial seasons for the
    # outbreak-feature plots; actual scoring uses the submission file's own dates).
    imdc_projection_epiweek_start: int = 41
    imdc_projection_epiweek_end: int = 40

    output_dir: Path = Path(".local/prototype_wis_analysis")

    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)
        self.convert_path_fields()


# ---------------------------------------------------------------------------
# Location-season discovery
# ---------------------------------------------------------------------------

def get_location_year_ids(cfg: ProgramConfig) -> list[tuple[str, int]]:
    """List (location_id, year) combos with a submission file on both sides."""
    pattern = "??_[0-9][0-9][0-9][0-9]"
    with_dirs = {Path(p).name for p in glob(str(cfg.with_update_dir / pattern))}
    no_dirs = {Path(p).name for p in glob(str(cfg.no_update_dir / pattern))}
    common = with_dirs & no_dirs

    missing = with_dirs ^ no_dirs
    if missing:
        print(f"Warning: {len(missing)} location-seasons only have predictions on one side: {sorted(missing)}")

    result = []
    for name in sorted(common):
        location_id, year_str = name.split("_")
        year = int(year_str)
        if cfg.use_location_ids and location_id not in cfg.use_location_ids:
            continue
        if cfg.use_projection_years and year not in cfg.use_projection_years:
            continue
        result.append((location_id, year))
    return result


# ---------------------------------------------------------------------------
# WIS (+ secondary metrics) scoring
# ---------------------------------------------------------------------------

def submission_to_quantile_frame(df: pd.DataFrame, cfg: ProgramConfig) -> pd.DataFrame:
    """Convert a submission-style DataFrame (index=date, columns=pred/lower_*/upper_*)
    into the (quantile, i_simulation)-by-date shape expected by `wis_score_vectorized`.
    """
    rows = {0.5: df[cfg.imdc_median_prediction_colname]}
    for level in cfg.imdc_required_predictive_intervals:
        alpha = 1.0 - level
        q_low, q_high = alpha / 2, 1.0 - alpha / 2
        lower_col = cfg.imdc_pi_colname_fmt_dict["lower"].format(level100=100 * level)
        upper_col = cfg.imdc_pi_colname_fmt_dict["upper"].format(level100=100 * level)
        rows[round(q_low, 6)] = df[lower_col]
        rows[round(q_high, 6)] = df[upper_col]

    quantile_df = pd.DataFrame(rows).T
    quantile_df.index = pd.MultiIndex.from_product(
        [quantile_df.index, [0]], names=["quantile", "i_simulation"]
    )
    return quantile_df


def score_location_season(pred_df: pd.DataFrame, obs_sr: pd.Series, cfg: ProgramConfig) -> dict | None:
    """Score one location-season's submission-style prediction against observations.

    Restricts scoring to weeks with a non-missing observation (the most recent
    season(s) may still be partially unobserved at the tail).
    """
    obs_aligned = obs_sr.reindex(pred_df.index)
    valid = obs_aligned.notna()
    n_total = len(pred_df)
    n_scored = int(valid.sum())
    if n_scored == 0:
        return None

    pred_valid = pred_df.loc[valid]
    obs_valid = obs_aligned.loc[valid]

    quantile_df = submission_to_quantile_frame(pred_valid, cfg)
    wis_arr = wis_score_vectorized(quantile_df, obs_valid)[0]  # shape (n_scored,)

    # NOTE: `coverages_vectorized` does `alphas or <default>` internally, which
    # raises on a non-empty numpy array (ambiguous truth value) -- pass a plain
    # list instead to sidestep that. Also round to avoid float noise (e.g.
    # 1.0 - 0.8 != 0.2 exactly) causing an exact-match quantile lookup to miss.
    alphas = [round(1.0 - level, 6) for level in cfg.imdc_required_predictive_intervals]
    cov_df = coverages_vectorized(
        quantile_df, obs_valid, alphas=alphas, calculate_coverage_likelihood=False
    )

    median = pred_valid[cfg.imdc_median_prediction_colname].to_numpy()
    obs_vals = obs_valid.to_numpy()
    err = median - obs_vals
    denom = np.abs(median) + np.abs(obs_vals)
    with np.errstate(invalid="ignore"):
        smape_vals = np.where(denom == 0, 0.0, 2 * np.abs(err) / denom)

    obs_mean = obs_vals.mean()
    result = dict(
        n_weeks_total=n_total,
        n_weeks_scored=n_scored,
        frac_scored=n_scored / n_total,
        wis_mean=float(np.mean(wis_arr)),
        # Scale-free variant, for averaging fairly across locations of very
        # different case-count magnitude (e.g. SP vs. AC).
        wis_norm=float(np.mean(wis_arr) / obs_mean) if obs_mean > 0 else np.nan,
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        smape=float(np.mean(smape_vals)),
    )
    for level in cfg.imdc_required_predictive_intervals:
        col = f"coverage_{level:0.3f}"
        result[f"coverage_{round(level * 100)}"] = float(cov_df[col].iloc[0])

    return result


def run_scoring(cfg: ProgramConfig, location_year_ids: list[tuple[str, int]]) -> pd.DataFrame:
    """Score every location-season under both scenarios into one long DataFrame."""
    obs_cache = DiseaseTimeSeriesCache(cfg.disease_cases_dir)
    scenario_dirs = {"with_update": cfg.with_update_dir, "no_update": cfg.no_update_dir}

    rows = []
    for location_id, year in location_year_ids:
        obs_sr = obs_cache.get_location(location_id)
        for scenario, base_dir in scenario_dirs.items():
            fpath = (
                base_dir
                / cfg.location_year_subdir_fmt.format(location_id=location_id, year=year)
                / cfg.submission_fname
            )
            if not fpath.exists():
                print(f"  Missing submission file (skipped): {fpath}")
                continue

            pred_df = pd.read_csv(fpath, parse_dates=["date"], index_col="date")
            result = score_location_season(pred_df, obs_sr, cfg)
            if result is None:
                print(f"  No observed weeks to score: {location_id} {year} ({scenario})")
                continue

            result.update(location_id=location_id, year=year, scenario=scenario)
            rows.append(result)

    return pd.DataFrame(rows)


def load_predictions_long(
        cfg: ProgramConfig, base_dir: Path, location_year_ids: list[tuple[str, int]]
) -> pd.DataFrame:
    """Load submission-style predictions for many location-seasons into one
    DataFrame indexed by (location_id, year, date). Mirrors the loading
    pattern in `revise_and_submit_projections.ipynb` (cell 5).
    """
    df_list, keys_list = [], []
    for location_id, year in location_year_ids:
        fpath = (
            base_dir
            / cfg.location_year_subdir_fmt.format(location_id=location_id, year=year)
            / cfg.submission_fname
        )
        if not fpath.exists():
            continue
        df_list.append(pd.read_csv(fpath, parse_dates=["date"], index_col="date"))
        keys_list.append((location_id, year))

    return pd.concat(df_list, keys=keys_list, names=["location_id", "year"])


def plot_predictive_timeseries_by_location(
        cfg: ProgramConfig, scores_df: pd.DataFrame, scenario: str,
        predictions_long_df: pd.DataFrame, out_pdf_path: Path,
):
    """One PDF, one page per location: full observed history overlaid with
    every season's median + 50%/95% prediction intervals, each season
    annotated with its mean WIS.
    """
    obs_cache = DiseaseTimeSeriesCache(cfg.disease_cases_dir)
    scenario_scores = (
        scores_df[scores_df["scenario"] == scenario]
        .set_index(["location_id", "year"])
    )

    location_ids = sorted(predictions_long_df.index.get_level_values("location_id").unique())

    with PdfPages(out_pdf_path) as pdf, plt.rc_context({
        "lines.linewidth": 1.5,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0,
        # "pdf.fonttype": 42,  # Troublesome on Ubuntu
    }):
        for location_id in location_ids:
            loc_df = predictions_long_df.loc[location_id]
            obs_sr = obs_cache.get_location(location_id)

            fig, ax = plt.subplots(figsize=(12, 4.5))

            for year in sorted(loc_df.index.get_level_values("year").unique()):
                year_df = loc_df.loc[year]

                ax.plot(year_df.index, year_df["pred"], color=PLOT_COLORS["median"])
                ax.fill_between(
                    year_df.index, year_df["lower_95"], year_df["upper_95"],
                    color=PLOT_COLORS["pi95"],
                )
                ax.fill_between(
                    year_df.index, year_df["lower_50"], year_df["upper_50"],
                    color=PLOT_COLORS["pi50"],
                )

                # --- Annotate this season's mean WIS, directly above it.
                # Anchored to a fixed height (axes fraction) rather than the
                # local PI peak, since very wide intervals (e.g. the
                # no-update scenario) would push the label off unpredictably.
                if (location_id, year) in scenario_scores.index:
                    row = scenario_scores.loc[(location_id, year)]
                    label = f"WIS={row['wis_mean']:0.0f}"
                    if row["frac_scored"] < 1:
                        label += "\n(partial)"
                    x_mid = year_df.index[0] + (year_df.index[-1] - year_df.index[0]) / 2
                    ax.annotate(
                        label, xy=(x_mid, 0.98), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=7,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.75),
                    )

            # --- Observations (full history)
            ax.plot(obs_sr, "s", color=PLOT_COLORS["obs"], label="Observed")

            ax.set_ylabel("Weekly Dengue Cases")
            ax.set_xlim(
                obs_sr.index.min(),
                loc_df.index.get_level_values("date").max() + pd.Timedelta(weeks=4),
            )
            ax.xaxis.set_major_locator(mpl.dates.YearLocator(base=1))
            ax.xaxis.set_minor_locator(mpl.dates.MonthLocator())

            scenario_title = "With outbreak-feature update" if scenario == "with_update" else "Without update"
            ax.set_title(f"{location_id} — {scenario_title}")

            handles = [
                mpl.lines.Line2D([0], [0], color=PLOT_COLORS["median"], label="Median Prediction"),
                mpl.patches.Patch(color=PLOT_COLORS["pi50"], label="50% Prediction Interval"),
                mpl.patches.Patch(color=PLOT_COLORS["pi95"], label="95% Prediction Interval"),
                mpl.lines.Line2D([0], [0], color=PLOT_COLORS["obs"], lw=0, marker="s", label="Observed Cases"),
            ]
            ax.legend(handles=handles, loc="upper left", fontsize=9)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def build_summary_tables(scores_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate per location-season scores into location/season/overall summaries."""
    metric_cols = [
        "wis_mean", "wis_norm", "mae", "rmse", "smape",
        "coverage_50", "coverage_80", "coverage_90", "coverage_95",
    ]
    tables = {}

    # --- Per (location, season): with vs. without side by side, plus delta
    pivot = scores_df.pivot_table(index=["location_id", "year"], columns="scenario", values=metric_cols)
    pivot.columns = [f"{metric}__{scenario}" for metric, scenario in pivot.columns]
    for metric in metric_cols:
        pivot[f"{metric}__delta_no_minus_with"] = (
            pivot[f"{metric}__no_update"] - pivot[f"{metric}__with_update"]
        )
    tables["by_location_season"] = pivot.sort_index()

    # --- Per location (averaged over seasons)
    tables["by_location"] = (
        scores_df.groupby(["location_id", "scenario"])[metric_cols].mean().unstack("scenario")
    )

    # --- Per season (averaged over locations)
    tables["by_season"] = (
        scores_df.groupby(["year", "scenario"])[metric_cols].mean().unstack("scenario")
    )

    # --- Overall (all location-seasons)
    tables["overall"] = scores_df.groupby("scenario")[metric_cols].mean()

    return tables


def plot_wis_comparison(scores_df: pd.DataFrame, out_pdf_path: Path):
    """Bar/scatter plots comparing WIS with vs. without the update."""
    with_color, no_color = "#49786c", "#b6603a"

    with PdfPages(out_pdf_path) as pdf:
        # --- Bar chart by location (mean over seasons)
        for metric, ylabel in [("wis_mean", "Mean WIS"), ("wis_norm", "Mean WIS / mean observed cases")]:
            by_loc = scores_df.groupby(["location_id", "scenario"])[metric].mean().unstack("scenario")
            by_loc = by_loc.sort_values("with_update", ascending=False)

            fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(by_loc)), 4))
            x = np.arange(len(by_loc))
            width = 0.38
            ax.bar(x - width / 2, by_loc["with_update"], width, label="With update", color=with_color)
            ax.bar(x + width / 2, by_loc["no_update"], width, label="Without update", color=no_color)
            ax.set_xticks(x)
            ax.set_xticklabels(by_loc.index, rotation=90, fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} by location (avg. over seasons)")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Bar chart by season (mean over locations)
        for metric, ylabel in [("wis_mean", "Mean WIS"), ("wis_norm", "Mean WIS / mean observed cases")]:
            by_season = scores_df.groupby(["year", "scenario"])[metric].mean().unstack("scenario")

            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(by_season))
            width = 0.38
            ax.bar(x - width / 2, by_season["with_update"], width, label="With update", color=with_color)
            ax.bar(x + width / 2, by_season["no_update"], width, label="Without update", color=no_color)
            ax.set_xticks(x)
            ax.set_xticklabels(by_season.index)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} by season (avg. over locations)")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Paired scatter: with vs. without, per location-season
        wide = scores_df.pivot_table(index=["location_id", "year"], columns="scenario", values="wis_norm")

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(wide["with_update"], wide["no_update"], alpha=0.6, color="#3b6ea5")
        lims = [0, max(wide["with_update"].max(), wide["no_update"].max()) * 1.05]
        ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("WIS/mean(obs) — with update")
        ax.set_ylabel("WIS/mean(obs) — without update")
        ax.set_title("Points above the diagonal: update improved WIS")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Outbreak-feature predictive distributions
# ---------------------------------------------------------------------------

def get_observed_feature_values(
        cfg: ProgramConfig, location_id: str, year: int, obs_sr: pd.Series
) -> tuple[pd.Series | None, bool]:
    """Compute actual outbreak-feature values from the observed weekly series,
    windowed the same way as the challenge submission (EW41 -> EW40 next year).

    Returns (feature_values, is_partial_season). `is_partial_season` is True
    when the observed series does not yet cover the full window (i.e. the
    computed features would be an under-count).
    """
    zero_date = year_week_to_date(year, cfg.dyn_zero_date_epiweek)
    window_start = year_week_to_date(year, cfg.imdc_projection_epiweek_start)
    window_end = year_week_to_date(year + 1, cfg.imdc_projection_epiweek_end)

    window_sr = obs_sr.loc[window_start:window_end]
    if window_sr.empty or window_sr.isna().all():
        return None, True

    is_partial = window_sr.index.max() < window_end

    single_row_df = window_sr.to_frame().T
    feats = outbreak_features_from_trajectories(
        single_row_df, zero_date=zero_date, feature_names=cfg.outbreak_feature_names
    )
    return feats.iloc[0], is_partial


def get_predicted_feature_samples(
        cfg: ProgramConfig, cache: OutbreakFeaturePredictionsCache,
        feature_name: str, location_id: str, year: int,
) -> pd.Series:
    """Predicted samples for one feature, with `peak_week` realigned to the
    dynamic model's own zero-date epiweek (see `ProgramConfig` docstring note).
    """
    samples = cache.get_sample_series(feature_name=feature_name, location_id=location_id, year=year)
    if feature_name == "peak_week":
        offset = cfg.dyn_zero_date_epiweek - cfg.feat_peak_ref_epiweek
        if offset:
            samples = samples - offset
    return samples


def plot_outbreak_feature_distributions(
        cfg: ProgramConfig, location_year_ids: list[tuple[str, int]], out_pdf_path: Path
):
    """One page per location-season: predicted-sample histograms for each
    outbreak feature, with the actual observed value marked.
    """
    feat_cache = OutbreakFeaturePredictionsCache(main_dir=cfg.outbreak_features_predictions_dir)
    for feature_name in cfg.outbreak_feature_names:
        df = feat_cache.load_file(feat_cache.get_file_path(feature_name=feature_name))
        feat_cache.add_df_to_cache(df, feature_name=feature_name)

    obs_cache = DiseaseTimeSeriesCache(cfg.disease_cases_dir)

    feature_labels = {
        "case_attack_rate": "Case attack rate (total cases)",
        "peak_week": "Peak week (weeks from EW41)",
        "peak_amplitude": "Peak weekly incidence",
    }

    with PdfPages(out_pdf_path) as pdf:
        for location_id, year in location_year_ids:
            obs_sr = obs_cache.get_location(location_id)
            observed_feats, is_partial = get_observed_feature_values(cfg, location_id, year, obs_sr)

            fig, axes = plt.subplots(
                1, len(cfg.outbreak_feature_names),
                figsize=(4 * len(cfg.outbreak_feature_names), 3.5),
            )
            for ax, feature_name in zip(axes, cfg.outbreak_feature_names):
                samples = get_predicted_feature_samples(cfg, feat_cache, feature_name, location_id, year)
                if samples.shape[0] == 0:
                    ax.set_title(f"{feature_name}\n(no prediction)")
                    continue

                ax.hist(samples, bins=30, density=True, color="#5b8dd6", alpha=0.75)

                if observed_feats is not None and feature_name in observed_feats.index:
                    val = observed_feats[feature_name]
                    label = "Observed" + (" (partial season)" if is_partial else "")
                    ax.axvline(val, color="#d64550", linestyle="--", linewidth=1.5, label=label)
                    ax.legend(fontsize=8)

                ax.set_xlabel(feature_labels.get(feature_name, feature_name))
                ax.set_ylabel("Density")

            fig.suptitle(f"{location_id} — season {year}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
