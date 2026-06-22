"""Utilities and helpers for calculating outbreak features from simulations
of the outbreak dynamics model.
"""
from typing import Union

import pandas as pd


def outbreak_features_from_trajectories(
        cases_df: pd.DataFrame,
        zero_date: pd.Timestamp,
        window_start: pd.Timestamp = None,
        window_end: pd.Timestamp = None,
        feature_names: list[str] = None
):
    """Calculate outbreak features from case trajectories.

    Parameters
    ----------
    cases_df : pd.DataFrame
        DataFrame of stochastic case trajectories, with the following expected
        signature:
        - Index: ID of the trajectories, can have one or multiple levels (agnostic).
        - Columns: Time points (e.g., weeks) of the outbreak, as datetime-like objects.
    zero_date: pd.Timestamp
        Reference date for temporal features. For example, peak week is
        calculated as number of weeks relative to this date.
    window_start : pd.Timestamp, optional
        Start of the time window for feature calculation. If None, uses the
        entire range of the DataFrame.
    window_end : pd.Timestamp, optional
        End of the time window for feature calculation. If None, uses the
        entire range of the DataFrame.

    feature_names : list of str, optional
        List of feature names to calculate. If None, calculates all available.

    Returns
    -------
    pd.DataFrame
        DataFrame of calculated outbreak features, with the following signature:
        - Index: Same as `cases_df` index (ID of the trajectories).
        - Columns: Calculated features, e.g., "case_attack_rate", "peak_week", "peak_amplitude".
    """
    df = cases_df
    if window_start and window_end:
        df = cases_df.loc[:, window_start:window_end]

    feature_names = feature_names or [
        "case_attack_rate",
        "peak_week",
        "peak_amplitude"
    ]

    # ---

    features_df = pd.DataFrame(index=df.index)
    # Signature: df.loc[(i_simulation, i_stochastic), feature]

    if "case_attack_rate" in feature_names:
        features_df["case_attack_rate"] = df.sum(axis=1)
    if "peak_week" in feature_names:
        features_df["peak_week"] = (df.idxmax(axis=1) - zero_date).dt.days / 7
    if "peak_amplitude" in feature_names:
        features_df["peak_amplitude"] = df.max(axis=1)

    return features_df


def outbreak_feature_stats(
        features_df: pd.DataFrame,
        groupby_var: Union[str, list[str]] = "i_simulation",
):
    """"""

    agg_funcs = {
        col: [
            ("mean", "mean"),
            ("std", "std"),
            ("q025", lambda x: x.quantile(0.025)),
            ("q975", lambda x: x.quantile(0.975)),
        ]
        for col in features_df.columns
    }

    feature_dists_df = (
        features_df
        .groupby(groupby_var, sort=False)
        .agg(agg_funcs)
    )

    feature_dists_df.columns.set_names(["feature", "stat"], inplace=True)

    return feature_dists_df
