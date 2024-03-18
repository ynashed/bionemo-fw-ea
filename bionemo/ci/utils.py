# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import concurrent
import math
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from bionemo.ci.columns import RequiredStatus


__all__: Sequence[str] = (
    # general utilties
    "JSON",
    "flatten1",
    # progress-bar related
    "ProgressBarLog",
    "to_tqdm_params",
    # HTTP and async utilities
    "execute",
    "http_get_all_pages",
    # dataframe utilities
    "ByStatus",
    "partition_status",
    "map_column_preserve_dtype",
    # date-manipulating utilities
    "safe_parse_date",
    "safe_to_date",
)

T = TypeVar('T')
A = TypeVar('A')
B = TypeVar('B')


# general utilties

JSON = Union[None, bool, str, float, int, List['JSON'], Dict[str, 'JSON']]
"""Type that represents every possible JSON value.
"""


def flatten1(xs: Iterable[Iterable[T]]) -> List[T]:
    """Expands an iterable-of-iterables into a single list of elements."""
    if isinstance(xs, str):
        return [xs]

    unrolled: List[T] = []
    for x in xs:
        if not isinstance(x, str):
            try:
                for sub in x:
                    unrolled.append(sub)
            except TypeError:
                unrolled.append(x)
        else:
            unrolled.append(x)
    return unrolled


# progress-bar related

ProgressBarLog = Dict[str, Any] | bool
"""Arguments for optionally confiuging a tqdm progress bar.

Either one of:
 - `True`: meaning that the progress bar should be created using default values
 - `False`: no progress bar printing
 - a `dict`: the configuration options for the progress bar; overrides defaults
"""


def to_tqdm_params(verbose: ProgressBarLog) -> Optional[Dict[str, Any]]:
    """Conveinent transformation into a `dict` for creating a progress bar or `None` for no progress bar.

    NOTE: Shalllow-copies the input iff it is a `dict`.
    """
    if isinstance(verbose, bool):
        if verbose:
            return {}
        else:
            return None
    else:
        return dict(**verbose)


# HTTP and async utilities


def http_get_all_pages(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    verbose: ProgressBarLog = False,
) -> Iterator[JSON]:
    """Performs a paginated HTTP GET on the URL: iteration yields additional network calls.

    An HTTP GET is sent to the :param:`url`, optionally also sending along the :param:`header` and :param:`params` too.

    The returned iterator allows the caller to obtain each response page from the server. Each call to `next` returns
    the currently obtained page and then invokes another HTTP GET call.

    The response header's `X-Total-Pages` indicates how many pages are in the total response, while `X-Next-Page`
    indicates which particular page a given response  corresponds to. Importantly, note that the outbound `page`
    parameter controls which page is being requested.

    Raises a `ValueError` if a non-200 HTTP OK status is received on any response.
    May raise other errors from the `requests` library's `get`.
    """
    tqdm_params = to_tqdm_params(verbose)
    more_to_obtain = True
    total: Optional[int] = None
    progress: Optional[tqdm] = None
    if params is None:
        params = {}

    while more_to_obtain:
        response = requests.get(url, headers=headers, params=params)
        if not response.ok:
            raise ValueError(f"Invalid request: {response}")

        if total is None and tqdm_params is not None:
            if 'X-Total-Pages' in response.headers:
                try:
                    total = int(response.headers['X-Total-Pages'])
                except:  # noqa
                    print(f"ERROR: X-Total-Pages is not an int! {response.headers=}")
                    raise
            else:
                total = 1

            if 'desc' not in tqdm_params:
                tqdm_params['desc'] = f"GET {url}"
            tqdm_params['total'] = int(total)
            progress = tqdm(**tqdm_params)

        if progress is not None:
            progress.update()

        yield response.json()

        if 'X-Next-Page' in response.headers:
            more_to_obtain = len(response.headers['X-Next-Page']) > 0
            params['page'] = response.headers['X-Next-Page']
        else:
            more_to_obtain = False


def execute(
    to_run: List[Callable[[], T]],
    n_threads: Optional[int],
    *,
    verbose: ProgressBarLog = False,
    timeout: Optional[int] = None,
    throw_first_exception: bool = False,
) -> List[T | Exception]:
    """Concurrently executes a list of functions using a threadpool.

    The functions to be executed, :param:`to_run`, should be I/O heavy. If they are compute heavy, then
    they will block each other's execution to due Python's GIL.

    The `n_threads` value controls how large the thread pool is for execution. A value of `None` means it will
    use the number of available CPU cores. Any non-`None` value must be a positive integer.

    The :param:`timeout` is specified in seconds. If specified, all executed functions must take less than this
    amount of time, otherwise a `TimeoutError` is raised. By default this value is `None`, meaning it will wait
    forever for completion.

    If :param:`throw_first_exception` is true, then the first exception raised by any of the executing functions
    is immediately re-raised to the caller. Note that if no exceptions are raised, then the returned list
    will have no `Exception` typed values in it.

    The returned list of results is in the same order as the input functions of :param:`to_run`. If the result
    is an `Exception`, then this was the raised by executing the function. Otherwise, the value is the result
    returned from the function.
    """
    if n_threads is not None and n_threads < 1:
        raise ValueError(f"When specified, number of threads must be >= 1, not {n_threads=}")

    tqdm_params = to_tqdm_params(verbose)
    if tqdm_params is not None:
        tqdm_params['total'] = len(to_run)
        progress = tqdm(**tqdm_params)
    else:
        progress = None

    if n_threads is None:
        results: List[T | Exception] = []
        for runnable in to_run:
            try:
                result = runnable()
                results.append(result)
            except Exception as error:  # noqa
                if throw_first_exception:
                    raise error
                results.append(error)
            if progress:
                progress.update()

        return result

    else:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_results_order: Dict[concurrent.futures.Future, int] = {
                executor.submit(runnable): i for i, runnable in enumerate(to_run)
            }
            for future in concurrent.futures.as_completed(future_results_order, timeout):
                if progress:
                    progress.update()
                index = future_results_order[future]
                try:
                    computed_value: T = future.result()
                    results.append((index, computed_value))
                except Exception as error:  # noqa
                    if throw_first_exception:
                        raise error
                    results.append((index, error))
        # return in original input order
        results.sort(key=lambda x: x[0])
        for i in range(len(results)):
            results[i] = results[i][1]  # type: ignore
        return results  # type: ignore


# dataframe utilities


class ByStatus(NamedTuple):
    idx_success: pd.Series
    idx_failed: pd.Series
    idx_other: pd.Series


def partition_status(pipeline_df: pd.DataFrame, *, cols: RequiredStatus = RequiredStatus()) -> ByStatus:
    """Create 3 row indices for successes, failures, and other on the status column."""
    idx_success = pipeline_df[cols.status] == 'success'
    idx_failed = pipeline_df[cols.status] == 'failed'
    return ByStatus(idx_success, idx_failed, idx_other=~(idx_success | idx_failed))


def map_column_preserve_dtype(column: pd.Series, f: Callable[[A], B]) -> pd.Series:
    """Apply a function to every value in a column & prevent Pandas from changing the type underneath the hood.


    Unfortunately, Pandas unexpectedly will silently change _some_ values during _some_ dataframe operations.
    It is known that Pandas will unhelpfully convert a datetime into a pandas.Timestamp unless the series' datatype
    is explicitly set to `object`. Unfortunately, when calling `map` on a series, there's
    """
    if not isinstance(column, pd.Series):
        raise TypeError(f"Can only operate on a pandas.Series, not a {type(column)} (column)")
    if len(column.values) == 0:
        raise ValueError("Must supply non-empty column!")
    values = [f(x) for x in column.values]
    if isinstance(values[0], str):
        expected_dtype: type = str
    elif isinstance(values[0], int):
        expected_dtype = int
    elif isinstance(values[0], float):
        expected_dtype = float
    elif isinstance(values[0], bool):
        expected_dtype = bool
    else:
        expected_dtype = object
    return pd.Series(values, dtype=expected_dtype, index=column.index)


def safe_parse_date(x: str | datetime | pd.Timestamp | np.datetime64) -> datetime:
    """Ensures that a date is a datetime instance.

    Parses an ISO-8601 formatted timestamp into a date if it is a string. No-op if the input is a datetime instance.
    Raises a `TypeError` otherwise.
    """
    if isinstance(x, str):
        return datetime.fromisoformat(x)
    elif isinstance(x, datetime):
        return x
    elif isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    elif isinstance(x, np.datetime64):
        return pd.to_datetime(x).to_pydatetime()
    else:
        raise TypeError(f"Expecting str in ISO format or datetime, but got {type(x)} ({x})")


def safe_to_date(x: Optional[str | datetime | pd.Timestamp | np.datetime64]) -> Optional[datetime]:
    """Ensures that the input is a date if it is not None. No-op if None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    else:
        return safe_parse_date(x)
