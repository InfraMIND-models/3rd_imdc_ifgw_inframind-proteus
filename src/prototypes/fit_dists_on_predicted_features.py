"""
Distribution fitting with statsmodels.
Fits Normal, Gamma, Log-Normal, Weibull, and Exponential distributions
to each (feature, location, year) combination. Collects AIC/BIC and KS
goodness-of-fit stats, and produces one subplot per distribution (with
correct histogram/PDF scaling).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
from plotly.subplots import make_subplots
from statsmodels.distributions.empirical_distribution import ECDF

# ---------------------------------------------------------------------------
# Distribution registry
# Each entry: (display_name, scipy.stats frozen-fitter callable)
# statsmodels doesn't have a unified .fit() for arbitrary parametric families,
# so we use scipy.stats under the hood but wrap the AIC/BIC/KS logic ourselves
# in a statsmodels-style way (log-likelihood + n_params → AIC/BIC).
# ---------------------------------------------------------------------------
DISTRIBUTIONS = {
    "normal":      (scipy.stats.norm,       2),   # loc, scale
    "gamma":       (scipy.stats.gamma,      3),   # a, loc, scale
    "lognormal":   (scipy.stats.lognorm,    3),   # s, loc, scale
    "weibull":     (scipy.stats.weibull_min, 3),  # c, loc, scale
    "exponential": (scipy.stats.expon,      2),   # loc, scale
}

COLORS = {
    # Assign a distinct base color per distribution; year variants get opacity
    "normal":      "#1f77b4",
    "gamma":       "#ff7f0e",
    "lognormal":   "#2ca02c",
    "weibull":     "#d62728",
    "exponential": "#9467bd",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_distribution(dist_obj, data):
    """Fit a scipy.stats continuous distribution and return params + stats."""
    params = dist_obj.fit(data)
    log_ll = np.sum(dist_obj.logpdf(data, *params))
    n = len(data)
    k = len(params)
    aic = 2 * k - 2 * log_ll
    bic = k * np.log(n) - 2 * log_ll
    ks_stat, ks_pval = scipy.stats.kstest(data, dist_obj.cdf, args=params)
    return {
        "params": params,
        "log_likelihood": log_ll,
        "aic": aic,
        "bic": bic,
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
    }


def param_names(dist_name, dist_obj):
    """Return human-readable parameter names for a distribution."""
    shapes = dist_obj.shapes.split(", ") if dist_obj.shapes else []
    return shapes + ["loc", "scale"]


def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_distribution_fitting(
    feat_predictions: dict,
    uf_table_df: pd.DataFrame,
    out_dir: Path = Path(".local/outbreak-feature-predictions/distributions"),
):
    """
    Parameters
    ----------
    feat_predictions : dict[str, pd.DataFrame]
        Keys are feature names. Each DataFrame must have columns:
        'location_id', 'year', and the feature column itself.
    uf_table_df : pd.DataFrame
        Must contain a 'uf' column with location identifiers.
    out_dir : Path
        Directory where HTML plots are saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    location_ids = list(uf_table_df["uf"].unique())
    n_dists = len(DISTRIBUTIONS)

    # Collect all goodness-of-fit records across everything
    all_gof_records = []

    for feature_name in feat_predictions.keys():
        print(f"\n{'='*60}")
        print(f"Feature: {feature_name}")

        for location_id in location_ids:
            df = feat_predictions[feature_name].copy()
            df = df[df["location_id"] == location_id]
            years = sorted(df["year"].unique())

            print(f"  Location: {location_id}  |  Years: {years}")

            # ------------------------------------------------------------------
            # Build figure: one subplot column per distribution
            # ------------------------------------------------------------------
            fig = make_subplots(
                rows=len(years),
                cols=n_dists,
                subplot_titles=[d.replace("_", "-").title() for d in DISTRIBUTIONS],
                shared_yaxes=False,
            )

            # Assign a distinct line style per year so overlapping years are
            # visually separable even without color (accessibility-friendly)
            dash_styles = ["solid", "dash", "dot", "dashdot", "longdash"]
            year_dash = {yr: dash_styles[i % len(dash_styles)] for i, yr in enumerate(years)}

            # ------------------------------------------------------------------
            # Fit each distribution for each year
            # ------------------------------------------------------------------
            for col_idx, (dist_name, (dist_obj, _n_params)) in enumerate(
                DISTRIBUTIONS.items(), start=1
            ):
                base_color = COLORS[dist_name]
                show_legend_for_dist = col_idx == 1  # only first subplot drives legend

                for yr_idx, year in enumerate(years):
                    year_df = df[df["year"] == year]
                    data = year_df[feature_name].dropna().values

                    if len(data) < 5:
                        print(f"    Skipping {dist_name}/{year}: too few data points ({len(data)})")
                        continue

                    # -- Fit
                    try:
                        result = fit_distribution(dist_obj, data)
                    except Exception as exc:
                        print(f"    Fit failed for {dist_name}/{year}: {exc}")
                        continue

                    params = result["params"]

                    # -- Store GoF record
                    pnames = param_names(dist_name, dist_obj)
                    param_dict = {f"param_{p}": v for p, v in zip(pnames, params)}
                    all_gof_records.append(
                        {
                            "feature": feature_name,
                            "location_id": location_id,
                            "year": year,
                            "distribution": dist_name,
                            "n": len(data),
                            "log_likelihood": result["log_likelihood"],
                            "aic": result["aic"],
                            "bic": result["bic"],
                            "ks_stat": result["ks_stat"],
                            "ks_pval": result["ks_pval"],
                            **param_dict,
                        }
                    )

                    # -- Histogram (only add once per year per subplot)
                    # Use np.histogram to get bin edges so the PDF x-range
                    # matches exactly → fixes the amplitude mismatch.
                    counts, bin_edges = np.histogram(data, bins="auto", density=True)
                    bin_width = bin_edges[1] - bin_edges[0]
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                    alpha = 0.25 + 0.15 * yr_idx  # slightly different opacity per year
                    fill_color = hex_to_rgba(base_color, alpha)
                    line_color = hex_to_rgba(base_color, min(alpha + 0.3, 1.0))

                    fig.add_trace(
                        go.Bar(
                            x=bin_centers,
                            y=counts,
                            width=bin_width * 0.9,
                            name=f"{year} (data)",
                            legendgroup=f"year_{year}",
                            showlegend=show_legend_for_dist,
                            marker=dict(
                                color=fill_color,
                                line=dict(color=line_color, width=1),
                            ),
                            opacity=0.7,
                        ),
                        row=yr_idx + 1,
                        col=col_idx,
                    )

                    # -- PDF curve: evaluated on the same x range as the data
                    x_pdf = np.linspace(
                        min(bin_edges), max(bin_edges), 300
                    )
                    y_pdf = dist_obj.pdf(x_pdf, *params)

                    # Label includes key GoF stats
                    label = (
                        f"{year} | AIC={result['aic']:.1f} "
                        f"KS p={result['ks_pval']:.3f}"
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=x_pdf,
                            y=y_pdf,
                            mode="lines",
                            name=label,
                            legendgroup=f"year_{year}",
                            showlegend=show_legend_for_dist,
                            line=dict(
                                color=hex_to_rgba(base_color, 0.95),
                                width=2,
                                dash=year_dash[year],
                            ),
                        ),
                        row=yr_idx + 1,
                        col=col_idx,
                    )

            # ------------------------------------------------------------------
            # Layout polish
            # ------------------------------------------------------------------
            fig.update_layout(
                title_text=(
                    f"Distribution fits — {feature_name} | location: {location_id}"
                ),
                height=280 * len(years),
                width=340 * n_dists,
                bargap=0.05,
                legend=dict(
                    title="Year (PDF label = AIC, KS p-value)",
                    orientation="v",
                    x=1.01,
                    y=1,
                ),
                template="plotly_white",
            )

            for col_idx in range(1, n_dists + 1):
                for yr_idx in range(len(years)):
                    fig.update_yaxes(title_text="Density", row=yr_idx + 1, col=col_idx)
                    fig.update_xaxes(title_text=feature_name, row=yr_idx + 1, col=col_idx)

            # ------------------------------------------------------------------
            # Save
            # ------------------------------------------------------------------
            html_path = out_dir / f"{feature_name}_{location_id}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            print(f"  → Saved: {html_path}")

    # --------------------------------------------------------------------------
    # Build GoF summary DataFrame
    # --------------------------------------------------------------------------
    gof_df = pd.DataFrame(all_gof_records)

    if not gof_df.empty:
        # Reorder columns for readability
        front_cols = [
            "feature", "location_id", "year", "distribution", "n",
            "log_likelihood", "aic", "bic", "ks_stat", "ks_pval",
        ]
        param_cols = [c for c in gof_df.columns if c.startswith("param_")]
        gof_df = gof_df[front_cols + param_cols]

        csv_path = out_dir / "goodness_of_fit.csv"
        gof_df.to_csv(csv_path, index=False)
        print(f"\nGoF summary saved → {csv_path}")
        print(gof_df.head(10).to_string(index=False))

    return gof_df


def load_data(
    predictions_dir = Path("predictions"),
    feature_names = None ,
    uf_table_fpath = Path("data/demographic/uf_table.csv")
):
    feature_names = feature_names or [
        "case_attack_rate",
        "peak_week",
        "peak_amplitude"
    ]

    feat_predictions = dict()
    #  d[feature_name] = feature_predictions_df
    for feature_name in feature_names:
        feat_predictions[feature_name] = pd.read_csv(predictions_dir / f"{feature_name}.csv")

    uf_table_df = pd.read_csv(uf_table_fpath)

    return feat_predictions, uf_table_df


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def compute_delta_aic(gof_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a delta_aic column: within each (feature, location_id, year) group,
    subtract the minimum AIC so the best model scores 0.
    """
    df = gof_df.copy()
    group_min = df.groupby(["feature", "location_id", "year"])["aic"].transform("min")
    df["delta_aic"] = df["aic"] - group_min
    return df


def best_distribution_per_feature(df_delta: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise which distribution wins overall for each feature.
    Returns a DataFrame with mean/median ΔAIC and win-rate per
    (feature, distribution), sorted by mean ΔAIC ascending.
    """
    records = []
    for feature, fdf in df_delta.groupby("feature"):
        n_groups = fdf.groupby(["location_id", "year"]).ngroups
        for dist, ddf in fdf.groupby("distribution"):
            win_rate = (ddf["delta_aic"] == 0).sum() / n_groups
            records.append({
                "feature":      feature,
                "distribution": dist,
                "mean_delta_aic":   ddf["delta_aic"].mean(),
                "median_delta_aic": ddf["delta_aic"].median(),
                "win_rate":         win_rate,
            })
    result = pd.DataFrame(records).sort_values(["feature", "mean_delta_aic"])
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_heatmap(df_delta: pd.DataFrame, feature: str) -> go.Figure:
    """ΔAIC heatmap: distribution × location (mean over years)."""
    fdf = df_delta[df_delta["feature"] == feature]
    pivot = (
        fdf.groupby(["distribution", "location_id"])["delta_aic"]
        .mean()
        .unstack("location_id")
    )
    # Order distributions by overall mean ΔAIC (best on top)
    order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[order]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=list(pivot.index),
            colorscale="RdYlGn_r",   # green = low ΔAIC = good
            colorbar=dict(title="Mean ΔAIC"),
            hoverongaps=False,
            hovertemplate="Location: %{x}<br>Distribution: %{y}<br>Mean ΔAIC: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Mean ΔAIC by distribution & location — {feature}",
        xaxis_title="Location",
        yaxis_title="Distribution",
        height=320 + 40 * len(pivot),
        template="plotly_white",
    )
    return fig


def _plot_boxplot(df_delta: pd.DataFrame, feature: str) -> go.Figure:
    """ΔAIC box plot: one box per distribution, pooled over all locations × years."""
    fdf = df_delta[df_delta["feature"] == feature]

    # Sort distributions by median ΔAIC so the best appears first
    order = (
        fdf.groupby("distribution")["delta_aic"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig = go.Figure()
    for dist in order:
        vals = fdf[fdf["distribution"] == dist]["delta_aic"]
        fig.add_trace(
            go.Box(
                y=vals,
                name=dist,
                marker_color=COLORS.get(dist, "#888"),
                boxmean="sd",       # show mean ± sd marker inside box
                line_width=1.5,
            )
        )

    # Reference lines for the "essentially equivalent" / "clear loser" thresholds
    for threshold, label, color in [
        (2,  "ΔAIC = 2 (equivalent)",  "rgba(0,150,0,0.4)"),
        (10, "ΔAIC = 10 (clear loser)", "rgba(200,0,0,0.4)"),
    ]:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="right",
        )

    fig.update_layout(
        title=f"ΔAIC distribution (pooled) — {feature}",
        yaxis_title="ΔAIC  (lower = better; 0 = best model in that group)",
        xaxis_title="Distribution",
        height=500,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def _plot_winrate(df_delta: pd.DataFrame, feature: str) -> go.Figure:
    """Bar chart: fraction of (location, year) groups where each distribution wins."""
    fdf = df_delta[df_delta["feature"] == feature]
    n_groups = fdf.groupby(["location_id", "year"]).ngroups

    win_counts = (
        fdf[fdf["delta_aic"] == 0]
        .groupby("distribution")
        .size()
        .reindex(COLORS.keys(), fill_value=0)
    )
    win_rates = win_counts / n_groups

    # Sort descending
    win_rates = win_rates.sort_values(ascending=False)

    fig = go.Figure(
        go.Bar(
            x=win_rates.index.tolist(),
            y=win_rates.values,
            marker_color=[COLORS.get(d, "#888") for d in win_rates.index],
            text=[f"{v:.0%}" for v in win_rates.values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1%} of groups<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Win rate (lowest AIC per group) — {feature}",
        yaxis=dict(title="Fraction of groups", tickformat=".0%", range=[0, 1.1]),
        xaxis_title="Distribution",
        height=420,
        template="plotly_white",
    )
    return fig


def compare_aic(
    gof_df: pd.DataFrame,
    out_dir: Path = Path(".local/outbreak-feature-predictions/dist_comparisons"),
) -> pd.DataFrame:
    """
    Run the full AIC comparison for all features in gof_df.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per (feature, distribution) with mean/median ΔAIC and win-rate.
        Also saved as 'aic_comparison_summary.csv' in out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_delta = compute_delta_aic(gof_df)
    summary_df = best_distribution_per_feature(df_delta)

    for feature in gof_df["feature"].unique():
        print(f"\n{'='*60}")
        print(f"Feature: {feature}")

        fig_heat = _plot_heatmap(df_delta, feature)
        fig_box  = _plot_boxplot(df_delta, feature)
        fig_win  = _plot_winrate(df_delta, feature)

        for fig, suffix in [
            (fig_heat, "aic_heatmap"),
            (fig_box,  "aic_boxplot"),
            (fig_win,  "aic_winrate"),
        ]:
            path = out_dir / f"{feature}_{suffix}.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            print(f"  → {path}")

        # Print a quick text summary for this feature
        feat_summary = summary_df[summary_df["feature"] == feature]
        print(feat_summary.to_string(index=False))

    csv_path = out_dir / "aic_comparison_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved → {csv_path}")

    return summary_df



# ---------------------------------------------------------------------------
# Entry point (replace with your actual objects)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    feat_predictions, uf_table_df = load_data()

    gof_df = run_distribution_fitting(
        feat_predictions=feat_predictions,   # your existing dict
        uf_table_df=uf_table_df,             # your existing DataFrame
    )

    print(gof_df)

    gof_df.to_csv(
        Path(".local/outbreak-feature-predictions/goodness_of_fit_summary.csv"),
        index=False
    )

    summary = compare_aic(gof_df)

    # Quick decision helper: best distribution per feature
    best = summary.loc[summary.groupby("feature")["mean_delta_aic"].idxmin()]
    print("\n=== Recommended distribution per feature ===")
    print(best[["feature", "distribution", "mean_delta_aic", "win_rate"]].to_string(index=False))


