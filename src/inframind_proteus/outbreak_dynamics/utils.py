"""Shared utilities for the outbreak dynamics module.

Includes:
- Timestamp parsing: ISO date string, float days, or YYYYWW epiweek integer
- RNG seed helpers
"""

from __future__ import annotations

import io
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple

import numpy as np
from epiweeks import Week as EpiWeek
import pandas as pd
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessPool
import yaml


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def year_week_to_date(year: int, week: int) -> pd.Timestamp | pd.NaT:
    """"""
    # return EpiWeek(year, week).startdate().isoformat()
    return pd.Timestamp(EpiWeek(year, week, system="cdc").startdate())


def epiweek_to_date(epiweek_int: int) -> pd.Timestamp | pd.NaT:
    """Convert a YYYYWW integer to the start date (Sunday) of that CDC epiweek.

    Parameters
    ----------
    epiweek_int:
        Integer of the form ``YYYYWW``, e.g. ``202340`` = week 40 of 2023.
        Uses the CDC epiweek convention (week starts on Sunday).

    Returns
    -------
    pd.Timestamp
        Sunday that opens the given CDC epiweek.
    """

    return year_week_to_date(year=epiweek_int // 100, week=epiweek_int % 100)
    # year =
    # week = epiweek_int % 100
    # return pd.Timestamp(EpiWeek(year, week, system="cdc").startdate())


def parse_timestamp(
    value: str | int | float | pd.Timestamp,
    zero_date: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Normalise a timestamp to a :class:`pandas.Timestamp`.

    Accepted formats:

    - ISO date string: ``"2023-10-02"``
    - YYYYWW epiweek integer: ``202340`` (→ Sunday opening that CDC epiweek)
    - Float days since ``zero_date``: ``0.0``, ``7.0``, …

    Parameters
    ----------
    value:
        Input timestamp in any of the accepted formats.
    zero_date:
        Reference date for float-based offsets.  Required when ``value``
        is a ``float``.

    Returns
    -------
    pd.Timestamp
    """
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, str) or isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, int):
        return epiweek_to_date(value)
    if isinstance(value, float):
        if zero_date is None:
            raise ValueError(
                "zero_date must be provided when value is a float (days offset)."
            )
        return zero_date + pd.Timedelta(days=value)
    raise TypeError(
        f"Unsupported timestamp type: {type(value).__name__!r}. "
        "Expected str (ISO date), int (YYYYWW), or float (days offset)."
    )


def load_yaml_dict(path: str | Path, safe=True) -> dict[Any, Any]:
    """Load a YAML file and return its contents as a dict."""
    loader = yaml.SafeLoader if safe else yaml.Loader
    with open(path) as fp:
        return yaml.load(fp, Loader=loader) or {}


def save_yaml_dict(data: dict[str, Any], path: str | Path, safe=True) -> None:
    """Save a dict to a YAML file."""
    dumper = yaml.SafeDumper if safe else yaml.Dumper
    with open(path, "w") as fp:
        yaml.dump(data, fp, Dumper=dumper)


def make_yaml_exportable_dict(
        data: dict,
        skip_keys: list = None,
        copy=True,
):
    """Converts some data types within a dictionary into other objects
    that can be read in a file (e.g. strings).
    Operates recursively through contained dictionaries.
    Changes are made inplace for all dictionaries.

    Parameters
    ----------
    data : dict
        The input dictionary to be converted.
    skip_keys : list, optional
        List of keys to skip during conversion. If a key is in this list, its
        value will not be converted, even if it is of a type that would normally
        be converted. Default is None, which means no keys are skipped.
    copy : bool, optional
        If True (default), the function will operate on a deep copy
        of the input dictionary. If False, the function returns the same
        dictionary, which is modified in place.
    """
    skip_keys = skip_keys or list()

    d = deepcopy(data) if copy else data

    for key, val in d.items():

        # Ordered and default dict
        if isinstance(val, (OrderedDict, defaultdict)):
            d[key] = dict(val)

        # pathlib.Path into its string
        if isinstance(val, Path):
            d[key] = str(val.expanduser())

        # Timestamps into string repr.
        if isinstance(val, pd.Timestamp):
            d[key] = str(val)

        # Specified iterables
        if isinstance(
                val, (tuple, np.ndarray)
        ):
            d[key] = list(val)

        # Recurse through inner dictionary
        if isinstance(val, dict):
            d[key] = make_yaml_exportable_dict(
                val, skip_keys=skip_keys, copy=False
            )

    return d


def add_set_argument(parser: ArgumentParser):
    """Append a `--set` argument to an ArgumentParser for nested
    configuration overrides.

    Modifies the provided `parser` in place to include a `--set` argument.
    """
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


def parse_set_arguments_with_yaml(set_args: list[list[str]]):
    """Parse a list of dot-notation key-value pairs into a nested dictionary.

    This function is designed to handle command-line arguments for setting
    nested configuration parameters. It takes a list of `[key, value]` pairs,
    where the key is a string using dot notation (e.g., 'a.b.c') to specify
    a path in a nested dictionary, and the value is a string representation
    of the data to be set at that path.

    The value string is parsed using YAML, which allows for automatic
    type inference of numbers, booleans, lists, and dictionaries.

    Parameters
    ----------
    set_args : list[list[str]]
        A list of two-element lists, where each inner list contains a
        dot-notation key and its corresponding string value.
        Example: `[['stage1.num_simulations', '8192'], ['sim_cfg.t0', '2022-01-01']]`

    Returns
    -------
    dict
        A nested dictionary representing the merged configuration overrides.

    Raises
    ------
    ValueError
        If a value string cannot be parsed by the YAML engine.
    """

    overrides = dict()
    for key, value in set_args:
        d = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            # Advance into the nested dictionary, creating if not there
            if part not in d:
                d[part] = dict()
            d = d[part]

        # Try to parse value with YAML engine
        try:
            # Use yaml.load to interpret types like int, float, bool, list
            parsed_value = yaml.load(
                io.StringIO(value),
                yaml.SafeLoader
            )
        except yaml.parser.ParserError:
            raise ValueError(
                f"Failed to parse --set argument for key '{key}': {value}"
            )

        # Add entry to the nested dictionary
        d[parts[-1]] = parsed_value

    return overrides


def apply_include_exclude_logic(
        all_series: pd.Series,
        include_list: list = None,
        exclude_list: list = None
):
    """Apply inclusion and exclusion logic to filter a series of values.

    This function applies optional inclusion and exclusion filters to determine
    which values from the input series should be used. If an inclusion list
    is provided, only those values are considered, and an error is raised if
    any requested value is not found in the input series. If no inclusion
    list is provided, all values in the input series are considered.
    An exclusion list, if present, is applied after inclusion logic.

    Parameters
    ----------
    all_series : pd.Series
        Series containing all possible values to filter from.
    include_list : list, optional
        List of values to include. If provided, only these values will be
        considered. If None, all values from all_series are considered.
    exclude_list : list, optional
        List of values to exclude from the final result. Applied after
        inclusion logic.

    Returns
    -------
    list
        The filtered list of values after applying inclusion and
        exclusion logic.

    Raises
    ------
    ValueError
        If a value specified in `include_list` is not found in `all_series`.
    """
    # --- Apply include logic
    if include_list:
        # Parameter `include_locations` was informed. Start from them.
        _requested_locs = include_list
        # Look for unrecognized locations
        for loc_name in _requested_locs:
            if loc_name not in all_series.values:
                raise ValueError(f"Location name {loc_name} not recognized in `locations_df`.")
    else:
        # Parameter `include_locations` None or empty. Start from all available.
        _requested_locs = list(all_series)

    # --- Apply exclude logic
    if exclude_list:
        use_locs = [state for state in _requested_locs if state not in exclude_list]
    else:
        use_locs = _requested_locs

    return use_locs


def map_parallel_or_sequential(
        task, *contents, ncpus=1, chunksize=1, pool=None
):
    """
    Apply a function (task) to a sequence of input parameters (contents),
    using either a sequential loop (if ncpus=1) or a pool of
    concurrent processes.
    Alternatively, for running in a pool of processes, a previously
    created pool can be informed, in which case ncpus is
    ignored.

    If `chunksize` > 1 and `ncpus` > 1, the parallel processes will be split
    into chunks of sequential execution, potentially reducing overhead
    costs of copying the data into subprocesses. For tasks that receive
    heavy data payloads, you can play with both `ncpus` and `chunksize`
    to achieve optimal performance.

    This function uses the `pathos` library to run processes in parallel.

    Parameters
    ----------
    task : callable
        A function to map on input contents.
    contents : iterable or multiple iterables
        Iterables containing sequences of inputs.
        Each argument must be an iterable or sequence of input arguments,
        which will then be passed to `task` in the same order as they are
        informed. If the iterables don't match in size, the size of the
        shortest one will be considered.
    ncpus : int
        Number of concurrent processes. If ncpus=1 (default), a simple for loop is used. If greater than 1, a process
        pool is created. If parameter `pool` is informed, ncpus is ignored.
    chunksize : int
        Size of each sequential chunk of execution.
    pool : Any
        A pathos process pool previously created. Overrides parameter 'ncpus' if informed.

    Returns
    -------
    results : list
        A list of results from applying `task` to all input items.
    """
    if chunksize > 1 and (ncpus > 1 or pool is not None):
        # --- Run parallel with chunking
        return map_parallel_or_sequential_chunks(
            task, *contents,
            ncpus=ncpus, chunksize=chunksize, pool=pool
        )

    if ncpus == 1 and pool is None:
        # --- Run sequentially
        results = list()
        for item in zip(*contents):
            results.append(task(*item))

        return results

    if pool is None:
        pool = ProcessPool(ncpus=ncpus)

    # --- Run non-chunked parallel
    return pool.map(task, *contents)


def map_parallel_or_sequential_chunks(
        task, *contents, ncpus=1, chunksize=1, pool=None
):
    """
    Apply a function (task) to a sequence of input parameters (contents)
    in batches (chunks), using either a sequential
    loop (if ncpus=1) or a pool of concurrent processes.

    Splitting between chunks may potentially reduce overhead costs of
    copying data into each subprocess.

    The input `contents` will be split into chunks of size `chunksize`.
    If `ncpus` > 1, each chunk is run in parallel through a
    process pool. Within each chunk, execution is sequential.

    Alternatively, for running in a pool of processes, a previously
    created pool can be informed, in which case `ncpus` is ignored.

    This function uses the `pathos` library to run processes in parallel.

    Parameters
    ----------
    task : callable
        A function to map on input contents.
    contents : iterable or multiple iterables
        Iterables containing sequences of inputs.
        Each argument must be an iterable or sequence of input arguments,
        which will then be passed to `task` in the same order as they are
        informed. If the iterables don't match in size, the size of the
        shortest one will be considered.
    ncpus : int
        Number of concurrent processes. If ncpus=1 (default), a simple
        for loop is used. If greater than 1, a process
        pool is created. If parameter `pool` is informed, ncpus is ignored.
    chunksize : int
        Size of each sequential chunk.
    pool : Any
        A pathos process pool previously created. Overrides parameter
        'ncpus' if informed.

    Returns
    -------
    results : list
        A list of results from applying `task` to all input items.


    Notes
    -----
    The `map_parallel_or_sequential` function calls
    `map_parallel_or_sequential_chunks`
    if both `ncpus` and `chunksize` are greater than one. Prefer to call
    it instead of this one, since it automatically chooses the appropriate
    formulation in each case.
    """
    num_items = len(list(zip(*contents)))
    num_chunks = (num_items - 1) // chunksize + 1

    # --- Build chunked lists of contents
    chunk_contents = list() # Holds each seq. of contents in chunks
    for single_contents in contents:
        chunk_contents.append(
            [single_contents[i * chunksize:(i+1) * chunksize] for i in range(num_chunks)]
        )

    # --- Define the chunkwise task as a sequential for loop
    def chunk_task(*_chunk):
        chunk_results = list()
        for _item in zip(*_chunk):
            chunk_results.append(task(*_item))
        return chunk_results

    # --- Run chunks sequentially
    if ncpus == 1:
        chunked_results = list()
        for chunk in zip(*chunk_contents):
            chunked_results.append(chunk_task(*chunk))

    # --- Run chunks through parallel processes
    else:
        if pool is None:
            pool = ProcessPool(ncpus=ncpus)
        chunked_results = pool.map(chunk_task, *chunk_contents)

    # --- Flatten the chunked results
    results = sum(chunked_results, list())

    return results


def make_axes_seq(
        num_axes, max_cols=3, total_width=9., ax_height=3.
) -> Tuple[plt.Figure, list[plt.Axes]]:
    """Create a 1D sequence of matplotlib `Axes` objects in figure, where
    axes are disposed in a grid of `max_cols` columns and as many rows
    as needed.

    Advantages of using this function instead of a call to
    `matplotlib.subplots` are:
    - You can directly specify the number of Axes objects instead of the
      numbers of rows and columns.
    - Axes objects are returned as a flat, 1D vector, in row-major order.
    - If `num_axes` is not a multiple of `max_cols`, the last row
     will contain empty spaces instead of unused Axes objects.

    Parameters
    ----------
    num_axes : int
        Total number of `Axes` objects to create in the figure.
    max_cols : int, optional
        Number of `Axes` objects in each row.
    total_width : float
        Width of the entire figure in default matplotlib units.
    ax_height : float
        Height of each Axes object in default matplotlib units.

    Returns
    -------
    fig : matplotlib.Figure
        A new figure containing the sequence of axes.
    axes : list[matplotlib.Axes]
        The 1D sequence of axes objects.
    """
    # Basic dependent numbers
    num_rows = (num_axes - 1) // max_cols + 1

    # Empty figure initialization and gridspec object (divides space into grids).
    fig = plt.figure(figsize=(total_width, num_rows * ax_height))
    gridspecs = fig.add_gridspec(num_rows, max_cols)

    # Creates the list of axes with the required number of axes
    axes = [fig.add_subplot(gridspecs[i]) for i in range(num_axes)]

    return fig, axes


def rotate_ax_labels(ax, angle=60, xy="x", which="major"):
    """
    Rotate tick labels of a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis whose tick labels will be rotated.
    angle : float, default=60
        The rotation angle in degrees.
    xy : {'x', 'y'}, default='x'
        Specifies whether to rotate x-axis or y-axis labels.
    which : {'major', 'minor', 'both'}, default='major'
        Specifies which tick labels to rotate.

    Returns
    -------
    None
        This function modifies the axis in place.
    """
    labels = (
        ax.get_xticklabels(which=which)
        if xy == "x" else ax.get_yticklabels(which=which))

    for label in labels:
        label.set(rotation=angle, horizontalalignment='right')

