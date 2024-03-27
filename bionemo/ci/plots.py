# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from bionemo.ci.columns import RequiredStartedAt, RequiredTimePlotCols
from bionemo.ci.utils import map_column_preserve_dtype, safe_parse_date


__all__: Sequence[str] = (
    'plot_count_by_week',
    "plot_pct_fail_by_week",
    "plot_duration_by_week",
    "arrange_weeks",
    "to_week",
)


def plot_count_by_week(
    df: pd.DataFrame, *, title: str, cols: RequiredTimePlotCols = RequiredTimePlotCols()
) -> plt.Figure:
    """Create a bar chart counting the number of rows per week.

    Each row in the input dataframe (:param:`df`) must have a `started_at` column. This column is an ISO-8601
    formatted timestamp. Rows whose started at date are in the same week are grouped together and their count is
    displayed in the returned bar chart.
    """
    _, counted = _counts(df, cols)

    fig, _ = plt.subplots()

    # get every week (according to 'started_at') in the data
    weeks = arrange_weeks(df, cols=cols)
    bar_plot = plt.bar(weeks, [counted[wk] for wk in weeks])

    plt.bar_label(bar_plot)
    plt.xticks(rotation=90)
    plt.title(title)

    plt.tight_layout()

    return fig


def _counts(df: pd.DataFrame, cols: RequiredTimePlotCols) -> Tuple[Dict[str, int], Dict[str, int]]:
    if len(df) == 0:
        raise ValueError("Need non-empty dataframe!")
    # defensive shallow copy
    df = df.copy()

    # group each CI failure by week & count
    counted_failure: Dict[str, int] = defaultdict(int)
    fail_df = df[df[cols.status] == 'failed']
    fail_df['year_week'] = fail_df[cols.started_at].map(to_week)
    for week, week_df in fail_df.groupby('year_week'):
        counted_failure[week] += len(week_df)

    # group each CI success by week & count
    counted_success: Dict[str, int] = defaultdict(int)
    success_df = df[df[cols.status] == 'success']
    success_df['year_week'] = success_df[cols.started_at].map(to_week)
    for week, week_df in success_df.groupby('year_week'):
        counted_success[week] += len(week_df)

    return counted_success, counted_failure


def plot_pct_fail_by_week(
    df: pd.DataFrame, *, title: str, cols: RequiredTimePlotCols = RequiredTimePlotCols()
) -> plt.Figure:
    """Create a bar chart counting the number of rows per week in 'failure' & dividing that by total count to get a % failing."""

    counted_success, counted_failure = _counts(df, cols)

    # get every week (according to 'started_at') in the data
    weeks = arrange_weeks(df, cols=cols)

    fail_by_pct: Dict[str, float] = {}
    for week in weeks:
        fail = counted_failure[week]
        success = counted_success[week]
        denom = fail + success
        if denom == 0:
            result = 0.0
        else:
            result = (fail / denom) * 100.0
        fail_by_pct[week] = result

    fig, _ = plt.subplots()

    bar_plot = plt.bar(
        weeks,
        [(fail_by_pct[wk] if wk in fail_by_pct else 0.0) for wk in weeks],
    )

    plt.bar_label(bar_plot, fmt="%.f")
    plt.xticks(rotation=90)
    plt.title(title)

    plt.tight_layout()

    return fig


def plot_duration_by_week(
    df: pd.DataFrame,
    *,
    title: str,
    rm_top: bool = True,
    show_point_times: bool = False,
    cols: RequiredTimePlotCols = RequiredTimePlotCols(),
) -> plt.Figure:
    """Create a bar chart showing the average & standard deviation for total elapsed time for CI pipeline completion, by week.

    Each row in the input dataframe (:param:`df`) must have a `total_elapsed_ts` column. This column is an ISO-8601
    formatted timestamp.
    """
    if len(df) == 0:
        raise ValueError("Need non-empty dataframe!")
    # defensive shallow copy
    df = df.copy()

    df[cols.created_at] = map_column_preserve_dtype(df[cols.created_at], safe_parse_date)
    df[cols.started_at] = map_column_preserve_dtype(df[cols.started_at], safe_parse_date)
    df[cols.finished_at] = map_column_preserve_dtype(df[cols.finished_at], safe_parse_date)

    # collect all total elapsed seconds for each week
    timing: Dict[str, List[int]] = defaultdict(list)
    df['year_week'] = df[cols.started_at].map(to_week)
    for week, week_df in df.groupby('year_week'):
        # can't use - on series because pandas changes the data type, which is both incredibly bad design and a blocker :(
        # it should be as easy as:
        #     elapsed = week_df[cols.finished_at] - week_df[cols.created_at]
        #     timing[week].extend(elapsed.map(lambda x: x.total_seconds()).values)
        # but pandas will very unhelpfully force these to be a numpy type, instead of the actual timedelta type that is defined
        # between two datetime instances
        for finished_at, created_at in zip(week_df[cols.finished_at].values, week_df[cols.created_at].values):
            elapsed: timedelta = finished_at - created_at
            assert isinstance(elapsed, timedelta)
            timing[week].append(elapsed.total_seconds())

    # get every week (according to 'started_at') in the data
    weeks = arrange_weeks(df, cols=cols)
    avg_elapsed_s_per_week: List[float] = []
    stddev_elapsed_s_per_week: List[float] = []
    for wk in weeks:
        elapsed_times = timing[wk]
        if len(elapsed_times) == 0:
            avg_elapsed_s_per_week.append(0.0)
            stddev_elapsed_s_per_week.append(0.0)
        else:
            avg = sum(elapsed_times) / len(elapsed_times)
            avg_elapsed_s_per_week.append(avg)
            if len(elapsed_times) == 1:
                stddev_elapsed_s_per_week.append(0.0)
            else:
                variance = sum([(e - avg) ** 2 for e in elapsed_times]) / (len(elapsed_times) - 1)
                stddev_elapsed_s_per_week.append(math.sqrt(variance))

    if rm_top:
        max_i = 0
        max_time = avg_elapsed_s_per_week[0]
        for i in range(1, len(avg_elapsed_s_per_week)):
            if avg_elapsed_s_per_week[i] > max_time:
                max_i = i
                max_time = avg_elapsed_s_per_week[i]
        del avg_elapsed_s_per_week[max_i]
        del stddev_elapsed_s_per_week[max_i]
        del weeks[max_i]

    # [math.log(x) if x > 0. else 0. for x in avg_elapsed_per_week],
    avg_elapsed_per_week = [x / 60.0 for x in avg_elapsed_s_per_week]
    stddev_elapsed_per_week = [x / 60.0 for x in stddev_elapsed_s_per_week]

    fig, ax = plt.subplots()

    # inspired from: https://stackoverflow.com/a/40020492/362021
    # plt.yticks(np.arange(min(avg_elapsed_per_week), max(avg_elapsed_per_week)+1, 100))
    plt.scatter(
        x=weeks,
        y=avg_elapsed_per_week,
    )
    plt.errorbar(
        x=weeks,
        y=avg_elapsed_per_week,
        yerr=stddev_elapsed_per_week,
        xerr=None,
        ls='none',
    )
    # adds the actual time to each data point
    if show_point_times:
        for i in range(len(weeks)):
            ax.annotate(f"{avg_elapsed_per_week[i]:1.0f}", (weeks[i], avg_elapsed_per_week[i]))

    plt.xticks(rotation=90)  # fontsize=???
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(250))
    plt.grid(True, which='major', axis='y', linewidth=1)
    plt.grid(True, which='minor', axis='y', linewidth=0.1)
    plt.title(title)
    plt.ylabel('Time (minutes)')
    plt.tight_layout()
    return fig


def arrange_weeks(df: pd.DataFrame, *, cols: RequiredStartedAt = RequiredStartedAt()) -> List[str]:
    """Like np.arrange, but outputs formatted week date strings for the range of started_at dates in the data."""
    first: datetime = df[cols.started_at].min()
    last: datetime = df[cols.started_at].max()

    weeks: List[str] = []
    current = first
    while current < last:
        weeks.append(to_week(current))
        current = current + timedelta(days=7)

    # maybe current was in the same week as last
    # so (current + 7days) > last
    # and thus we *don't* need to add last!
    if last.isocalendar().week != current.isocalendar().week:
        weeks.append(to_week(last))

    return weeks


def to_week(x: datetime) -> str:
    """Converts a datetime into the year & week number.

    Format for the returned string is `YYYY-MMM-wkWW` where `YYYY` is the year, 'MMM' is the 3-letter month,
    and `WW` is the week number. Weeks are numbered from `01` to `52`, inclusive.
    """
    return f"{x.year}-{x.strftime('%b').lower()}-wk{x.isocalendar().week}"
