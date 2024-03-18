# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

import pandas as pd

from bionemo.ci.columns import (
    RequiredCiJobColumns,
    RequiredCompletionCols,
    RequiredDateCols,
    RequiredElapsedTimeCols,
    RequiredPipelineCols,
    RequiredSha,
    RequiredStageCols,
    RequiredTimeCols,
    RequiredTotalDurationCols,
    check_cols_adhere_contract,
    columns_of,
    raise_value_error_if_missing_columns,
)
from bionemo.ci.gitlab_api import (
    GitLabHeaders,
    api_pipelines_for_project,
    http_get_bridge_trigger_for_pipelines,
    http_get_details_for_pipelines,
    http_get_stages_for_pipelines,
)
from bionemo.ci.utils import map_column_preserve_dtype, partition_status, safe_parse_date, safe_to_date


__all__: Sequence[str] = (
    # main metrics: how does CI fail on a project?
    "ci_pipeline_runs_for_project",
    # obtain a project's CI pipeline stages/jobs and the stages that were triggered
    # in _other_ project's CI pipelines as a result of the project's pipeline
    "df_http_get_all_ci_jobs_and_bridges",
    # and then combine all of these into a single set of stages associated with the project's pipeline ID
    "combine_pipeline_stages",
    # support:
    # filter-through a project's CI pipeline information to obtain only
    # one pipeline that corresponds to a single commit
    "single_pipeline_per_commit_event",
    "consolodate_by_sha",
    "last_completed",
    # support:
    # calculate and record the total elapsed time (created to complete), duration (time spent executing), and queued time (before executing)
    # for each pipeline: these are supposed to encompass the entire set of pipeline stages (i.e. after `combine_pipeline_stages`)
    "Time",
    "calculate_pipeline_durations",
    "total_elapsed_time",
    "merge_to_single_row_per_pipeline_total_duration",
)


@dataclass(frozen=True)
class ProjectCiResults:
    failed_ci_pipeline_run_df: pd.DataFrame
    """Contains failure_reason and stage information for all failed pipelines.
    """

    describe_all_ci_runs_df: pd.DataFrame
    """Contains all pipeline runs (no stage information) on a project. Includes success and failures.
    """


def ci_pipeline_runs_for_project(
    headers: GitLabHeaders,
    project_id: str,
    branch: str = 'dev',
    cache: Optional[Path] = None,
    verbose: bool = True,
) -> ProjectCiResults:
    """Performs all Gitlab API calls and assembles a project's CI pipeline failure & success history.

    Only selects CI pipelines from the :param:`project_id` ID with commits on the specified :param:`branch`.

    By default, this function will always perform new HTTP GET calls to the Gitlab API to get the latest data.
    However, one may supply a non-`None` :param:`cache` directory to save the responses from the API calls.
    Subsequent invocations of this function will re-use these cached responses.

    :param:`verbose` controls whether or not progress bars are displayed on STDOUT.
    """
    # NOTE: We are obtaining all CI pipeline information for the entire project.
    pipeline_df = None
    if cache is not None:
        try:
            pipeline_df = pd.read_csv(str(cache / 'pipeline_df.csv'))
        except:  # noqa
            pass
        else:
            print(f"Loaded cached {cache}/pipeline_df.csv")
    if pipeline_df is None:
        all_pipelines_for_project = api_pipelines_for_project(headers, project_id, verbose=verbose)
        pipeline_df = pd.DataFrame.from_records(all_pipelines_for_project)
        if cache is not None:
            pipeline_df.to_csv(str(cache / 'pipeline_df.csv'), index=False)
            print(f"Saved cached {cache}/pipeline_df.csv")

    # NOTE: We are narrowing this down to pipelines that ran on the specified branch
    #       that were tied to some commit-pushing event.

    pipeline_df = single_pipeline_per_commit_event(pipeline_df, branch=branch, source='push')

    idx_success, idx_failed, _ = partition_status(pipeline_df)

    # NOTE: Here, we have narrowed down even further to only the pipelines that had status=failed.
    failed_pipeline_df = pipeline_df[idx_failed]
    print(
        f"Out of {len(pipeline_df)} commits to {branch=}, {sum(idx_failed)} "
        f"commits failed CI and {sum(idx_success)} succeeded."
    )
    print(f"CI for stable {branch=} branch has a *SUCCESS* rate of {(sum(idx_success) / len(pipeline_df)*100.):0.2f}%")

    # NOTE: Getting details on all executed pipelines in the project's branch.
    details_for_pipelines_df = None
    if cache is not None:
        try:
            details_for_pipelines_df = pd.read_csv(str(cache / 'details_for_pipelines_df.csv'))
        except:  # noqa
            pass
        else:
            print(f"Loaded cached {cache}/details_for_pipelines_df.csv")
            details_for_pipelines_df['started_at'] = map_column_preserve_dtype(
                details_for_pipelines_df['started_at'], safe_to_date
            )
    if details_for_pipelines_df is None:
        details_for_pipelines_df = details_for_pipelines_df = df_http_get_details_for_pipelines(
            headers, project_id, pipeline_ids=pipeline_df['id'].values, del_ref=True, verbose=verbose
        )
        if cache is not None:
            details_for_pipelines_df.to_csv(str(cache / "details_for_pipelines_df.csv"), index=False)
            print(f"Saved cached {cache}/details_for_pipelines_df.csv")

    # NOTE: These will have all of the individual pipeline run stages
    #       Some stages will have staus=success. However, there will be at least one
    #       stage with status=failed for any pipeline_id in these two dataframes.
    jobs_for_failed_df: Optional[pd.DataFrame] = None
    jet_stages_when_failed_df: Optional[pd.DataFrame] = None
    if cache is not None:
        try:
            jobs_for_failed_df = pd.read_csv(str(cache / 'jobs_for_failed_df.csv'))
            jet_stages_when_failed_df = pd.read_csv(str(cache / 'jet_stages_when_failed_df.csv'))
        except:  # noqa
            pass
        else:
            print("Loaded cached jobs_for_failed_df and jet_stages_when_failed_df")
    if jobs_for_failed_df is None or jet_stages_when_failed_df is None:
        jobs_for_failed_df, jet_stages_when_failed_df = df_http_get_all_ci_jobs_and_bridges(
            headers,
            project_id,
            pipeline_ids=failed_pipeline_df['id'].values,
            verbose=verbose,
        )
        if cache is not None:
            jobs_for_failed_df.to_csv(str(cache / 'jobs_for_failed_df.csv'), index=False)
            jet_stages_when_failed_df.to_csv(str(cache / 'jet_stages_when_failed_df.csv'), index=False)
            print('Cached jobs_for_failed_df and jet_stages_when_failed_df')

    idx_jet_success_fail, idx_jet_fail_fail, _ = partition_status(jet_stages_when_failed_df)

    # ...maybe...use failed_pipeline_df instead?
    print(
        f"Whenever {branch=} CI failed and jet ran, jet FAILED {sum(idx_jet_fail_fail)} times "
        f"({(sum(idx_jet_fail_fail) / len(jet_stages_when_failed_df))*100.:0.2f}%)"
    )
    print(
        f"Whenever {branch=} CI failed and jet ran, jet SUCCEEDED {sum(idx_jet_success_fail)} times "
        f"({(sum(idx_jet_success_fail) / len(jet_stages_when_failed_df))*100.:0.2f}%)"
    )

    # NOTE: We stack these two dataframes together, giving us all of the actual CI stages (bridge and w/in project)
    #       that ran. This information lets us see each stage's runtime, whether or not it failed, and if it failed,
    #       the reason that Gitlab CI recorded for it.
    combined_pipeline_stages_df = combine_pipeline_stages(jet_stages_when_failed_df, jobs_for_failed_df)

    pipeline_durations = calculate_pipeline_durations(combined_pipeline_stages_df)

    # NOTE: Here, we only consider stages that failed.
    _, idx_failed_combined, _ = partition_status(combined_pipeline_stages_df)
    failed_only_combined_pipeline_stages_df = combined_pipeline_stages_df[idx_failed_combined]

    # NOTE: We want one row for each failed pipeline with the stage and duration information.
    #
    #       For each pipeline, there can be more than one failing stage.
    #       This is because the CI pipeline is a DAG, not a sequence. Since we want to obtain a single row
    #       for each failing CI pipeline, we need to merge rows from our CI stage dataframe.
    #
    #       We combine rows w/ the same pipeline ID, using a separator (+) to aggregate the values that occur for
    #       failure reason and stage name. We also calculate the entire pipeline's duration (s) and put that in
    #       each row.
    ci_pipeline_run_df = merge_to_single_row_per_pipeline_total_duration(
        failed_only_combined_pipeline_stages_df, pipeline_durations
    )

    # NOTE: Add started_at column to failed CI pipeline reasons dataframe.
    idx = details_for_pipelines_df['pipeline_id'].isin(set(ci_pipeline_run_df['pipeline_id'].values))
    selected_df = details_for_pipelines_df[idx][['pipeline_id', 'started_at', 'sha']]

    ci_pipeline_run_df: pd.DataFrame = (
        ci_pipeline_run_df.set_index('pipeline_id')
        .join(selected_df.set_index('pipeline_id'), on='pipeline_id')
        .reset_index()
    )

    return ProjectCiResults(
        failed_ci_pipeline_run_df=ci_pipeline_run_df,
        describe_all_ci_runs_df=details_for_pipelines_df,
    )


def single_pipeline_per_commit_event(
    pipeline_df: pd.DataFrame,
    *,
    branch: str = 'dev',
    source: str = 'push',
    cols: RequiredPipelineCols = RequiredPipelineCols(),
) -> pd.DataFrame:
    """Ensures that each commit is tied to exactly one CI pipeline run.

    Resolves ties by selecting the commit that has the most recent completed CI run.
    """
    raise_value_error_if_missing_columns(RequiredPipelineCols, cols, pipeline_df)

    pipeline_df = pipeline_df[pipeline_df[cols.ref] == branch]
    if len(pipeline_df) == 0:
        raise ValueError(f"{branch=} not found in {len(pipeline_df)} pipeline datas: {set(pipeline_df[cols.ref])=}")

    pushes_df = pipeline_df[pipeline_df[cols.source] == source]
    if len(pushes_df) == 0:
        raise ValueError(f"{source=} not found in {len(pushes_df)} pipeline datas: {set(pushes_df[cols.source])=}")

    commits = set(pushes_df[cols.sha])
    if len(pushes_df) == len(commits):
        print(f"Each {source=} on {branch=} has a distinct commit")
        return pushes_df

    else:
        print(f"Found {len(commits)} distinct commits but there are {len(pushes_df)} " f"{source} events for {branch}")
        if len(commits) > len(pushes_df):
            raise ValueError(
                "More commits than push events should not be possible! "
                f"Found {len(commits)} commits and {len(pushes_df)} pipeline events to {branch=}"
            )

        print("Selecting one pipeline run per commit")
        return consolodate_by_sha(pushes_df, cols=cols)


def consolodate_by_sha(pushes_df: pd.DataFrame, *, cols: RequiredSha = RequiredSha()) -> pd.DataFrame:
    """Replaces rows that have the same sha value with the one that was marked as completed last."""
    raise_value_error_if_missing_columns(RequiredSha, cols, pushes_df)
    if len(pushes_df) == 0:
        raise ValueError("Must supply non-empty dataframe!")
    resolved_rows: List[pd.Series] = []
    for _, df in pushes_df.groupby(cols.sha):
        if len(df) == 1:
            resolved_rows.append(df.iloc[0])
        else:
            resolved_rows.append(last_completed(df))
    return pd.DataFrame.from_records(resolved_rows)


def last_completed(df: pd.DataFrame, *, cols: RequiredCompletionCols = RequiredCompletionCols()) -> pd.Series:
    """Selects the row that was last completed.

    Specifically, a row that has status as either 'failed' or 'success' and whose 'created_at' date is the most recent.
    """
    raise_value_error_if_missing_columns(RequiredCompletionCols, cols, df)
    if len(df) < 1:
        raise ValueError(f"{len(df)=} must be positive int")
    df = df.copy()
    completed_df = df[df[cols.status].isin(['failed', 'success'])]
    # convert the timestamp string into a datetime object, which has an ordering
    completed_df[cols.created_at] = map_column_preserve_dtype(completed_df[cols.created_at], safe_parse_date)
    return completed_df.sort_values(by=cols.created_at).iloc[-1]


class CiPipelineStageDfs(NamedTuple):
    """For a specific project's CI pipeline: contains all of the project's direct CI stage information
    as well as CI stage information from _other_ project's CI pipelines that are triggered from the original
    project's pipeline definition.
    """

    ci_stages_df: pd.DataFrame
    jet_stages_df: pd.DataFrame


def df_http_get_details_for_pipelines(
    headers: GitLabHeaders,
    project_id: str,
    pipeline_ids: Sequence[str],
    *,
    cols: RequiredDateCols = RequiredDateCols(),
    del_ref: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Use the Gitlab API to get details for all supplied pipelines."""
    details_for_pipelines_df = pd.DataFrame.from_records(
        http_get_details_for_pipelines(
            headers,
            n_threads=os.cpu_count(),
            project_id=project_id,
            pipeline_ids=pipeline_ids,
            rename_pipeline_id=True,
            verbose=verbose,
        )
    )
    raise_value_error_if_missing_columns(RequiredDateCols, cols, details_for_pipelines_df)

    # remove redundant columns
    del details_for_pipelines_df[cols.project_id]
    # often we call w/ all ref=<one branch>
    if del_ref:
        del details_for_pipelines_df[cols.ref]

    # convert strings to datetime objects

    details_for_pipelines_df[cols.created_at] = map_column_preserve_dtype(
        details_for_pipelines_df[cols.created_at], safe_to_date
    )
    details_for_pipelines_df[cols.started_at] = map_column_preserve_dtype(
        details_for_pipelines_df[cols.started_at], safe_to_date
    )
    details_for_pipelines_df[cols.finished_at] = map_column_preserve_dtype(
        details_for_pipelines_df[cols.finished_at], safe_to_date
    )

    return details_for_pipelines_df


def df_http_get_all_ci_jobs_and_bridges(
    headers: GitLabHeaders, project_id: str, pipeline_ids: Sequence[str], *, verbose: bool = True
) -> CiPipelineStageDfs:
    """Uses the Gitlab API to get the CI jobs and the jet-trigger stages for all supplied pipelines."""
    jdf, fails = http_get_bridge_trigger_for_pipelines(
        headers,
        n_threads=os.cpu_count(),
        project_id=project_id,
        pipeline_ids=pipeline_ids,
        verbose=verbose,
        insert_pipeline_id=True,
    )
    jet_stages_df = pd.DataFrame.from_records(jdf)
    del jdf
    if verbose:
        if len(fails) == 0:
            print("All pipelines had a jet-trigger")
        else:
            print(f"There were {len(fails)} / {len(pipeline_ids)} pipelines that did not have a jet-trigger stage")

    ci_stages_df = pd.DataFrame.from_records(
        http_get_stages_for_pipelines(
            headers,
            n_threads=os.cpu_count(),
            project_id=project_id,
            pipeline_ids=pipeline_ids,
            verbose=verbose,
            insert_pipeline_id=True,
        )
    )
    if 'Unnamed: 0' in ci_stages_df:
        del ci_stages_df['Unnamed: 0']
    if verbose:
        print(f"There are {len(ci_stages_df)} individual job stages (non-bridge)")

    return CiPipelineStageDfs(ci_stages_df=ci_stages_df, jet_stages_df=jet_stages_df)


def combine_pipeline_stages(
    jet_stages_when_failed_df: pd.DataFrame,
    jobs_for_failed_df: pd.DataFrame,
    *,
    cols: RequiredStageCols = RequiredStageCols(),
) -> pd.DataFrame:
    """Combines the bridge-triggered stages along with a project's original CI pipeline stages."""
    raise_value_error_if_missing_columns(RequiredStageCols, cols, jet_stages_when_failed_df)
    raise_value_error_if_missing_columns(RequiredStageCols, cols, jobs_for_failed_df)

    common_columns_stages = columns_of(RequiredStageCols, cols)
    jet_slim_df = jet_stages_when_failed_df[common_columns_stages]
    jobs_slim_df = jobs_for_failed_df[common_columns_stages]

    combined_pipeline_stages_df = pd.concat([jobs_slim_df, jet_slim_df], ignore_index=True)

    return combined_pipeline_stages_df


@dataclass(frozen=True)
class Time:
    duration: timedelta
    """Time spent executing the pipeline."""
    elapsed: timedelta
    """Time spent from when the pipeline was created until when it was complete."""
    queued: timedelta
    """Time spent waiting for the pipeline to execute."""


def calculate_pipeline_durations(
    combined_pipeline_stages_df: pd.DataFrame,
    *,
    cols: RequiredTimeCols = RequiredTimeCols(),
) -> Dict[str, Time]:
    """Groups the combined CI stage information by pipeline to calculate the total duration & elapsed time of all stages.

    The returned dictionary's keys are pipeline IDs and the values are the CI timing information.
    """
    raise_value_error_if_missing_columns(RequiredTimeCols, cols, combined_pipeline_stages_df)

    pipeline_durations = {}
    for pid, df in combined_pipeline_stages_df.groupby(cols.pipeline_id):
        total_duration_seconds = df[cols.duration].sum()
        total_queued_seconds = df[cols.queued_duration].sum()
        pipeline_durations[pid] = Time(
            duration=timedelta(seconds=int(total_duration_seconds)),
            elapsed=total_elapsed_time(df, cols=cols, validate=False),
            queued=timedelta(seconds=int(total_queued_seconds)),
        )

    return pipeline_durations


def total_elapsed_time(
    df: pd.DataFrame,
    *,
    cols: RequiredElapsedTimeCols = RequiredElapsedTimeCols(),
    validate: bool = True,
) -> timedelta:
    """Calculate the total elapsed time (completed_at - created_at) across all pipeline stages in the dataframe."""
    if validate:
        raise_value_error_if_missing_columns(RequiredElapsedTimeCols, cols, df)

    # defensive shallow copy to prevent mutation of input
    df = df.copy()

    # only consider completed stages so that finished_at is defined
    df = df[df[cols.status].isin(['failed', 'success'])]

    time_df = df[[cols.created_at, cols.finished_at]]
    time_df[cols.created_at] = map_column_preserve_dtype(time_df[cols.created_at], safe_parse_date)
    time_df[cols.finished_at] = map_column_preserve_dtype(time_df[cols.finished_at], safe_parse_date)

    earliest_stage_created: datetime = time_df[cols.created_at].min()
    latest_stage_finished: datetime = time_df[cols.finished_at].max()

    elapsed: timedelta = latest_stage_finished - earliest_stage_created
    return elapsed


def merge_to_single_row_per_pipeline_total_duration(
    failed_pipeline_metadata_df: pd.DataFrame,
    pipeline_durations: Dict[str, timedelta],
    *,
    cols_i: RequiredCiJobColumns = RequiredCiJobColumns(),
    cols_a: RequiredTotalDurationCols = RequiredTotalDurationCols(),
    combine_char: str = " + ",
) -> pd.DataFrame:
    """Produces a dataframe where each row corresponds to exactly one CI pipeline run.

    The input dataframe must describe all of the CI stage runs for a given branch.
    Stages that were ran in the same pipeline will have the same pipeline_id. This function
    combines rows that have the same pipeline_id using the :param:`combine_char`. The columns
    kept are the pipeline_id, stage, name, failure_reason, and total_duration_s. These are
    specified to specific values with the :params:`cols_i` input.

    Additionally, this function adds new columns to the output dataframe that capture the
    pipeline's total time: queued, duration, and elapsed. These added columns are specified
    by the :param:`cols_a` columns schema.
    """
    raise_value_error_if_missing_columns(RequiredCiJobColumns, cols_i, failed_pipeline_metadata_df)
    check_cols_adhere_contract(RequiredTotalDurationCols, cols_a)

    if len(pipeline_durations) == 0:
        raise ValueError("Cannot supply empty pipeline durations.")

    if len(failed_pipeline_metadata_df) == 0:
        raise ValueError("Cannot supply empty dataframe.")

    def stringify_and_combine(values) -> str:
        return combine_char.join([str(x) for x in values])

    # building the dataframe
    pipeline_failure_reason_stage_rows = []
    for pipeline_id, df in failed_pipeline_metadata_df.groupby(cols_i.pipeline_id):
        # we've aggregated over pipeline_id
        # so df = all jobs for a unique pipeline run

        try:
            pipeline_timing_info = pipeline_durations[pipeline_id]
        except KeyError as e:
            raise ValueError(
                f"No duration information for {pipeline_id=}. "
                "These are calculated from different data than what is provided!"
            ) from e

        pipeline_failure_reason_stage_rows.append(
            {
                cols_i.pipeline_id: pipeline_id,
                cols_i.stage: stringify_and_combine(df[cols_i.stage].values),
                cols_i.name: stringify_and_combine(df[cols_i.name].values),
                cols_i.failure_reason: stringify_and_combine(df[cols_i.failure_reason].values),
                cols_a.total_duration_s: pipeline_timing_info.duration.seconds,
                cols_a.total_duration_ts: pipeline_timing_info.duration,
                cols_a.total_elapsed_s: pipeline_timing_info.elapsed.seconds,
                cols_a.total_elapsed_ts: pipeline_timing_info.elapsed,
                cols_a.total_queued_s: pipeline_timing_info.queued.seconds,
                cols_a.total_queued_ts: pipeline_timing_info.queued,
            }
        )

    pipeline_failure_reason_stage_df = pd.DataFrame.from_records(pipeline_failure_reason_stage_rows)

    return pipeline_failure_reason_stage_df
