"""Shared utilities for the outbreak dynamics module.

Includes:
- Timestamp parsing: ISO date string, float days, or YYYYWW epiweek integer
- RNG seed helpers
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
from pathos.multiprocessing import ProcessPool
import yaml


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def epiweek_to_date(epiweek_int: int) -> pd.Timestamp:
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
    from epiweeks import Week

    year = epiweek_int // 100
    week = epiweek_int % 100
    return pd.Timestamp(Week(year, week, system="cdc").startdate())


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


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as fp:
        return yaml.safe_load(fp)


def save_yaml_dict(data: dict[str, Any], path: str | Path) -> None:
    """Save a dict to a YAML file."""
    with open(path, "w") as fp:
        yaml.safe_dump(data, fp)


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
