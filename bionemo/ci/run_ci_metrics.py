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
import warnings
from pathlib import Path
from typing import Optional, Sequence

import click

from bionemo.ci.pipeline_stats import ci_pipeline_runs_for_project
from bionemo.ci.plots import plot_count_by_week, plot_duration_by_week, plot_pct_fail_by_week


__all__: Sequence[str] = ()


@click.command()
@click.option("--project", required=True, default='65301', type=str, show_default=True, help="Gitlab project ID.")
@click.option("--branch", required=True, default='dev', type=str, show_default=True, help="Branch in repository.")
@click.option(
    '--cache', required=False, type=str, help="If supplied, directory path to use for caching Gitlab API responses."
)
@click.option(
    "--output",
    required=True,
    default='.output_ci_metrics/',
    type=str,
    show_default=True,
    help="Location of output directory.",
)
def entrypoint(project: str, branch: str, cache: Optional[str], output: str) -> None:
    if 'GITLAB_TOKEN' not in os.environ:
        raise EnvironmentError("Missing GITLAB_TOKEN in environment!")

    print(f"Gitlab Project ID:            {project}")
    print(f"Repository branch:            {branch}")
    print(f"Cache dir for API responses?: {cache}")
    print(f"Output directory location?:   {output}")
    print('-' * 80)

    def check_dir(name, flag):
        if name is not None:
            d: Optional[Path] = Path(name).absolute()
            if d.is_file():
                raise ValueError(f"Invalid {flag} directory: {name} is a file!")
            d.mkdir(parents=True, exist_ok=True)
            return d
        else:
            return None

    cache_dir = check_dir(cache, '--cache')
    output_dir = check_dir(output, '--output')

    project_ci_pipeline_results = ci_pipeline_runs_for_project(
        headers={"PRIVATE-TOKEN": os.environ['GITLAB_TOKEN']},
        project_id=project,
        branch=branch,
        cache=cache_dir,
        verbose=True,
    )

    # CI failures: reason + stage information
    pipeline_failure_reason_stage_df = project_ci_pipeline_results.failed_ci_pipeline_run_df
    output_metrics_failed_pipelines_csv = output_dir / "pipeline_failure_reason_stage_df.csv"
    print(f"Saving CI pipeline with failure reasons dataframe to CSV: {output_metrics_failed_pipelines_csv}")
    pipeline_failure_reason_stage_df.to_csv(str(output_metrics_failed_pipelines_csv), index=False)
    print(pipeline_failure_reason_stage_df.head())
    print('-' * 80)

    # All CI pipeline information: successes & failures
    details_for_pipelines_df = project_ci_pipeline_results.describe_all_ci_runs_df
    output_non_failed_pipelines_csv = output_dir / "details_for_pipelines_df.csv"
    print(f"Saving CI pipeline successes and pending dataframe to CSV: {output_non_failed_pipelines_csv}")
    details_for_pipelines_df.to_csv(str(output_non_failed_pipelines_csv), index=False)
    print(details_for_pipelines_df.head())
    print('-' * 80)

    # fix up formatting -- this is because pandas will silently our data w/o warning or reason!
    complete_all_ci_runs_df = project_ci_pipeline_results.describe_all_ci_runs_df.copy()
    complete_all_ci_runs_df = complete_all_ci_runs_df[~complete_all_ci_runs_df['finished_at'].isna()]
    complete_all_ci_runs_df = complete_all_ci_runs_df[~complete_all_ci_runs_df['started_at'].isna()]
    complete_all_ci_runs_df = complete_all_ci_runs_df[~complete_all_ci_runs_df['created_at'].isna()]
    # only care about completed
    complete_all_ci_runs_df = complete_all_ci_runs_df[complete_all_ci_runs_df['status'].isin(['success', 'failed'])]

    # failure count by week
    plot_weekly_ci_failures = plot_count_by_week(
        complete_all_ci_runs_df,
        title="Weekly Count of Stable Branch BioNeMo Framework CI Failures",
    )
    output_plot = output_dir / "weekly_ci_failures_on_stable_branch.svg"
    print(f"Saving weekly plot of CI failures on stable branch ('dev'): {output_plot}")
    plot_weekly_ci_failures.savefig(str(output_plot), format='svg')

    # duration by week

    plot_elapsed_times = plot_duration_by_week(
        complete_all_ci_runs_df,
        title="Weekly Average Elapsed Time for CI Pipeline Completion",
        rm_top=True,  # for one identified outlier that makes the whole plot crazy
        show_point_times=False,  # nice detail, but gets cluttered fast
    )
    output_time_plot = output_dir / "weekly_avg_elapsed_time.svg"
    print(f"Saving weekly plot of average elapsed time for CI pipeline completion to: {output_time_plot}")
    plot_elapsed_times.savefig(str(output_time_plot), format='svg')

    # % fail by week
    plot_pct_fail = plot_pct_fail_by_week(
        complete_all_ci_runs_df,
        title="Weekly % CI Failures on Stable Branch BioNeMo Framework",
    )
    output_pct_plot = output_dir / "weekly_pct_ci_fail.svg"
    print(rf"Saving weekly plot of % of CI pipeline failures to: {output_pct_plot}")
    plot_pct_fail.savefig(str(output_pct_plot), format='svg')


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    entrypoint()
    # Notes on future work:
    # -> report: what are the stages that fail the most
    # -> report: what is the SLA on how often it takes a pipeline to run?
    # -> report: what is the SLA on how often a pipeline on dev fails? *why can't this be trending towards 100%?
