"""Utilities and helpers for calculating outbreak features from simulations
of the outbreak dynamics model.
"""
from pathlib import Path
from typing import Union, Iterable

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


class OutbreakFeaturePredictionsCache:
    """Represent predictions from the outbreak features models into the
    outbreak dynamics ecosystem.
    """
    _samples_by_feature: dict[str, pd.DataFrame]
    # Contains predictions as samples for each location and year, concatenated
    # into a single data frame.

    feature_names = [
            "case_attack_rate",
            "peak_week",
            "peak_amplitude",
    ]
    # Default feature names expected. Canbe overridden on __init__.

    def __init__(
            self,
            feature_names: list[str] = None,
            peak_ref_epiweek: int = 40,
            # Used by the outbreak features model as the "zero" for peak week

            main_dir: Union[str, Path] = (
                    Path("outputs/validation_round_outbreak_features")
            ),
            # fname_fmt: str = "{feature_name}_{location_id}.csv",  # -()- By feature and location
            fname_fmt: str = "{feature_name}.csv",  # By feature only
    ):
        self.feature_names = feature_names or self.feature_names
        self.peak_ref_epiweek = peak_ref_epiweek

        self.main_dir = Path(main_dir)
        self.fname_fmt = fname_fmt

        # --- Initialize an empty buffer for each feature
        # No index in cache, all variables are columns
        self._samples_by_feature = dict()

    def get_file_path(
            self, **kwargs
    ):
        """Prepare path to a file containing outbreak feature predictions.

        Parameters
        ----------
        location_id : str
        year : int
        feature_name : str
        """
        return self.main_dir / self.fname_fmt.format(**kwargs)

    @staticmethod
    def _append_fpath_info(fpath: Union[str, Path, None], msg: str) -> str:
        if fpath is not None:
            msg += f" (File: {fpath})"
        return msg

    def validate_samples_dataframe(
            self, df: pd.DataFrame,
            fpath: Union[str, Path, None] = None
    ):
        """"""
        # --- Identifying variables (required)
        id_vars = [
            "location_id",
            "year",
            "i_sample",
        ]
        _missing = list()
        for var in id_vars:
            if var not in df.columns:
                _missing.append(var)
        if _missing:
            msg = (
                f"DataFrame is missing required identifying variables '{_missing}'. "
                f"Expected variables within these: {id_vars}. "
            )
            self._append_fpath_info(fpath, msg)
            raise ValueError(msg)

        # # --- Columns
        # [ NO VALIDATION FOR OTHER COLUMNS - allow for extra unknown columns ]
        # _strange_cols = list()
        # for col in df.columns:
        #     if col not in self.feature_names:
        #         _strange_cols.append(col)
        # if len(_strange_cols) > 0:
        #     msg = (
        #         f"DataFrame contains unexpected columns '{_strange_cols}'. "
        #         f"Expected columns within these: {self.feature_names}. "
        #     )
        #     self._append_fpath_info(fpath, msg)
        #     raise ValueError(msg)


    def load_file(
            self,
            fpath: Union[str, Path]
    ) -> pd.DataFrame:
        """Load a CSV file containing outbreak feature predictions.

        Parameters
        ----------
        fpath : Union[str, Path]
            Path to the CSV file containing outbreak feature predictions.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the loaded outbreak feature predictions.

        Notes
        -----
        **Expected signature/variables of the CSV file:**

        - `location_id`: Identifier for the location (e.g., region, city).
        - `year`: Start year of the period covered by the outbreak features.
        - `i_sample`: Id of the sample (unique for each location and year).
        - `[_feature_name]`: Columns for each outbreak feature, e.g.,
           "case_attack_rate", "peak_week", "peak_amplitude".

        """
        try:
            df = pd.read_csv(
                fpath,
            )
        except Exception as e:
            # Add file path to message
            msg = (
                f"Error loading CSV file '{fpath}': {e}. "
            )
            raise ValueError(msg) from e
        self.validate_samples_dataframe(df, fpath=fpath)

        return df

    def add_df_to_cache(self, df: pd.DataFrame, feature_name: str):
        """Add a DataFrame of outbreak feature predictions to the cache.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing outbreak feature predictions to add to the cache.
            Must have the same signature as expected for outbreak feature predictions.
        feature_name : str
            Name of the outbreak feature corresponding to the DataFrame being added.

        Raises
        ------
        ValueError
            If the DataFrame does not have the expected signature or if the feature name is not recognized.
        """
        if feature_name not in self.feature_names:
            raise ValueError(
                f"Feature name '{feature_name}' is not recognized. "
                f"Expected one of: {self.feature_names}."
            )
        self.validate_samples_dataframe(df)
        if feature_name in self._samples_by_feature:
            # Concatenate to existing entry
            self._samples_by_feature[feature_name] = pd.concat(
                [self._samples_by_feature[feature_name], df],
                axis=0,
                ignore_index=True,
                sort=False,
            )
        else:
            # Create new entry
            self._samples_by_feature[feature_name] = df

    def get_sample_series(
            self,
            feature_name: str,
            location_id,
            year: int,
    ) -> pd.Series:
        df = self._samples_by_feature[feature_name]

        sr = (
            df
            .set_index(["location_id", "year", "i_sample"])
            .xs(location_id)
            .xs(year)
            [feature_name]
        )

        assert sr.index.name == "i_sample"
        assert isinstance(sr, pd.Series)

        if sr.shape[0] == 0:
            print(
                "Warning: Empty outbreak prediction series for: "
                f"{feature_name=}, {location_id=}, {year=}"
            )

        return sr
