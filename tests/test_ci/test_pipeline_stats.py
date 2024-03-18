# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime, timedelta
from functools import partial

import pandas as pd
from pytest import fixture, raises

from bionemo.ci.columns import (
    RequiredCiJobColumns,
    RequiredElapsedTimeCols,
    RequiredPipelineCols,
    RequiredSha,
    RequiredStageCols,
    RequiredTimeCols,
    RequiredTotalDurationCols,
    columns_of,
    raise_value_error_if_missing_columns,
)
from bionemo.ci.pipeline_stats import (
    calculate_pipeline_durations,
    combine_pipeline_stages,
    consolodate_by_sha,
    last_completed,
    merge_to_single_row_per_pipeline_total_duration,
    single_pipeline_per_commit_event,
    total_elapsed_time,
)


@fixture(scope='module')
def now() -> datetime:
    return datetime.now()


@fixture(scope='module')
def elapsed_time_cols() -> RequiredElapsedTimeCols:
    return RequiredElapsedTimeCols()


@fixture(scope='module')
def time_df(now: datetime, elapsed_time_cols: RequiredElapsedTimeCols) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        {
            elapsed_time_cols.status: pd.Series(['success', 'ignored', 'failed']),
            elapsed_time_cols.created_at: pd.Series(
                [now, now - timedelta(seconds=100), now + timedelta(seconds=10)], dtype='object'
            ),
            elapsed_time_cols.finished_at: pd.Series(
                [
                    now + timedelta(seconds=5),
                    now + timedelta(seconds=100),
                    now + timedelta(seconds=20),
                ],
                dtype='object',
            ),
        }
    )
    raise_value_error_if_missing_columns(RequiredElapsedTimeCols, elapsed_time_cols, df)
    return df


@fixture(scope='module')
def pipeline_cols() -> RequiredPipelineCols:
    return RequiredPipelineCols()


@fixture(scope='module')
def pipeline_df(now: datetime, pipeline_cols: RequiredPipelineCols) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        {
            pipeline_cols.status: pd.Series(["success", 'failed', 'success']),
            pipeline_cols.created_at: pd.Series(
                [now, now - timedelta(seconds=100), now + timedelta(seconds=10)], dtype='object'
            ),
            pipeline_cols.source: pd.Series(['push'] * 3),
            pipeline_cols.ref: pd.Series(['main'] * 3),
            pipeline_cols.sha: pd.Series(["0", "0", "1"]),
        }
    )
    raise_value_error_if_missing_columns(RequiredPipelineCols, pipeline_cols, df)
    return df


@fixture(scope='module')
def stage_cols() -> RequiredStageCols:
    return RequiredStageCols()


@fixture(scope='module')
def stage_df(now: datetime, stage_cols: RequiredStageCols) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        {
            stage_cols.pipeline_id: pd.Series(['1', '2']),
            stage_cols.failure_reason: pd.Series(['', '']),
            stage_cols.queued_duration: pd.Series([10, 20]),
            stage_cols.duration: pd.Series([30, 40]),
            stage_cols.finished_at: pd.Series(
                [now + timedelta(seconds=100), now + timedelta(seconds=200)], dtype='object'
            ),
            stage_cols.started_at: pd.Series([now + timedelta(seconds=10), now + timedelta(seconds=21)]),
            stage_cols.created_at: pd.Series([now, now + timedelta(seconds=1)]),
            stage_cols.status: pd.Series(['success', 'success']),
            stage_cols.name: pd.Series(['test', 'build']),
            stage_cols.stage: pd.Series(['something', 'here']),
        }
    )
    raise_value_error_if_missing_columns(RequiredStageCols, stage_cols, df)
    return df


@fixture(scope='module')
def jobs_df(stage_df: pd.DataFrame, stage_cols: RequiredStageCols) -> pd.DataFrame:
    df = stage_df.copy(deep=True)
    df[stage_cols.pipeline_id] = pd.Series(['3', '4'])
    raise_value_error_if_missing_columns(RequiredStageCols, stage_cols, df)
    return df


@fixture(scope='module')
def time_cols() -> RequiredTimeCols:
    return RequiredTimeCols()


@fixture(scope='module')
def duration_df(now: datetime, time_cols: RequiredTimeCols) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            time_cols.finished_at: pd.Series(
                [now + timedelta(seconds=11), now + timedelta(seconds=14)], dtype='object'
            ),
            time_cols.created_at: pd.Series([now, now + timedelta(seconds=1)], dtype='object'),
            time_cols.queued_duration: pd.Series([1, 2], dtype=int),
            time_cols.status: pd.Series(['success', 'success']),
            time_cols.duration: pd.Series([10, 11], dtype=int),
            time_cols.pipeline_id: pd.Series(["100", "101"]),
        }
    )
    raise_value_error_if_missing_columns(RequiredTimeCols, time_cols, df)
    return df


@fixture(scope='module')
def ci_job_cols() -> RequiredCiJobColumns:
    return RequiredCiJobColumns()


@fixture(scope='module')
def total_time_cols() -> RequiredTotalDurationCols:
    return RequiredTotalDurationCols()


def test_single_pipeline_per_commit_event(pipeline_df: pd.DataFrame, pipeline_cols: RequiredPipelineCols):
    # no rows w/ specified branch (ref)
    with raises(ValueError):
        bad_pipeline_df = pipeline_df.copy(deep=True)
        bad_pipeline_df[pipeline_cols.ref] = "invalid_branch"
        single_pipeline_per_commit_event(bad_pipeline_df, cols=pipeline_cols, branch="main")

    # no rows w/ specified source
    with raises(ValueError):
        bad_pipeline_df = pipeline_df.copy(deep=True)
        bad_pipeline_df[pipeline_cols.source] = "invalid_source"
        single_pipeline_per_commit_event(bad_pipeline_df, cols=pipeline_cols, source="push")

    assert len(pipeline_df) == 3
    df = single_pipeline_per_commit_event(pipeline_df, branch='main', source='push', cols=pipeline_cols)
    assert len(df) == 2
    assert set(df[pipeline_cols.sha].values) == {"0", "1"}
    assert (df[pipeline_cols.status] == 'success').all()

    idempodent = single_pipeline_per_commit_event(df, branch='main', source='push', cols=pipeline_cols)
    assert len(idempodent) == 2
    assert set(idempodent[pipeline_cols.sha].values) == {"0", "1"}
    assert (idempodent[pipeline_cols.status] == 'success').all()


def test_consolodate_by_sha(pipeline_df: pd.DataFrame, pipeline_cols: RequiredPipelineCols):
    # wrong columns
    with raises(ValueError):
        consolodate_by_sha(pd.DataFrame({'hello': [120, 240], "world": [60, 30]}), cols=pipeline_cols)

    # empty
    with raises(ValueError):
        consolodate_by_sha(pd.DataFrame(columns=columns_of(RequiredSha, pipeline_cols)), cols=pipeline_cols)

    df = consolodate_by_sha(pipeline_df, cols=pipeline_cols)
    assert len(df) == 2
    assert set(df[pipeline_cols.sha].values) == {"0", "1"}


def test_last_completed(now: datetime, time_df: pd.DataFrame, elapsed_time_cols: RequiredElapsedTimeCols):
    xs = {c: [] for c in elapsed_time_cols.columns()}
    invalid_df = pd.DataFrame(xs)
    with raises(ValueError):
        last_completed(invalid_df)
    invalid_df = time_df.copy()
    del invalid_df[elapsed_time_cols.status]
    with raises(ValueError):
        last_completed(invalid_df)

    row = last_completed(time_df, cols=elapsed_time_cols)
    actual = row[elapsed_time_cols.created_at]
    assert isinstance(actual, datetime)
    assert actual == (now + timedelta(seconds=10))


def test_combine_pipeline_stages(stage_df: pd.DataFrame, jobs_df: pd.DataFrame, stage_cols: RequiredStageCols):
    with raises(ValueError):
        combine_pipeline_stages(pd.DataFrame(), jobs_df, cols=stage_cols)
    with raises(ValueError):
        combine_pipeline_stages(stage_df, pd.DataFrame(), cols=stage_cols)
    df = combine_pipeline_stages(stage_df, jobs_df, cols=stage_cols)
    assert len(df) == 4


def test_calculate_pipeline_durations(duration_df: pd.DataFrame, time_cols: RequiredTimeCols):
    with raises(ValueError):
        calculate_pipeline_durations(pd.DataFrame(), cols=time_cols)
    time_by_pipeline = calculate_pipeline_durations(duration_df, cols=time_cols)
    assert isinstance(time_by_pipeline, dict)
    assert len(time_by_pipeline) == 2


def test_total_elapsed_time(time_df: pd.DataFrame, elapsed_time_cols: RequiredElapsedTimeCols):
    with raises(ValueError):
        invalid_df = time_df.copy()
        del invalid_df[elapsed_time_cols.status]
        total_elapsed_time(invalid_df, cols=elapsed_time_cols)

    duration = total_elapsed_time(time_df, cols=elapsed_time_cols)
    assert isinstance(duration, timedelta)
    assert duration.seconds == 20


def test_merge_to_single_row_per_pipeline_total_duration(
    stage_df: pd.DataFrame,
    ci_job_cols: RequiredCiJobColumns,
    total_time_cols: RequiredTotalDurationCols,
    time_cols: RequiredTimeCols,
):
    merge_fn = partial(merge_to_single_row_per_pipeline_total_duration, cols_i=ci_job_cols, cols_a=total_time_cols)

    # invalid columns
    with raises(ValueError):
        merge_fn(pd.DataFrame(), {})
    # empty durations
    with raises(ValueError):
        merge_fn(pd.DataFrame(columns=stage_df.columns), {})
        merge_to_single_row_per_pipeline_total_duration(stage_df, {})

    durations = calculate_pipeline_durations(stage_df, cols=time_cols)
    # empty dataframe
    with raises(ValueError):
        merge_fn(pd.DataFrame(columns=stage_df.columns), durations)

    merged_df = merge_fn(stage_df, durations)

    raise_value_error_if_missing_columns(RequiredTotalDurationCols, total_time_cols, merged_df)
    # won't have all columns -- notably missing 'status', 'started_at', and 'duration' in output
    assert ci_job_cols.pipeline_id in merged_df.columns
    assert ci_job_cols.stage in merged_df.columns
    assert ci_job_cols.name in merged_df.columns
    assert len(merged_df) == 2
