# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import partial
from typing import List, Optional, Sequence, Tuple, TypedDict

import requests

from bionemo.ci.utils import JSON, ProgressBarLog, execute, flatten1, http_get_all_pages


__all__: Sequence[str] = (
    "GitLabHeaders",
    # Gitlab API: direct 1-1 correspondence with REST API (https://docs.gitlab.com/ee/api/pipelines.html)
    "api_pipelines_for_project",
    "api_bridges_for_pipeline",
    "api_stages_for_pipeline",
    "api_pipeline_details",
    # Functions calling multiple Gitlab APIs for a single purpose.
    "http_get_bridge_trigger_for_pipelines",
    "http_get_stages_for_pipelines",
    "http_get_details_for_pipelines",
)

GitLabHeaders = TypedDict(
    "GitLabHeaders",
    {
        'PRIVATE-TOKEN': str,
    },
)
"""The required headers for the Gitlab API.

Notably, all Gitlab API calls require a user authentiation token in the HTTP header.
"""

#
#
# Gitlab API: direct 1-1 correspondce with REST API (https://docs.gitlab.com/ee/api/pipelines.html)
#
#


def api_pipelines_for_project(
    headers: GitLabHeaders,
    project_id: str,
    *,
    verbose: ProgressBarLog = False,
) -> List[JSON]:
    """HTTP GET all of a project's CI pipelines using the Gitlab API.

    HTTP GET /projects/:id/pipelines in the Gitlab API docs.

    :param:`verbose` controls whether or not a progress bar is printed to stdout.
    """
    return flatten1(
        http_get_all_pages(
            f"https://gitlab-master.nvidia.com/api/v4/projects/{project_id}/pipelines/",
            headers=headers,
            verbose=verbose,
        )
    )


def api_bridges_for_pipeline(
    headers: GitLabHeaders, project_id: str, pipeline_id: str, *, verbose: ProgressBarLog = False
) -> List[JSON]:
    """HTTP GET all bridges for a specific CI pipeline.

    HTTP GET /projects/:id/pipelines/:pipeline_id/bridges in the Gitlab API docs.

    Bridges are CI stages that are defined in other project's CI pipelines. They are triggered in the other
    project (using its permissions, settings, etc.) and linked to the :param:`pipeline_id`.

    :param:`verbose` controls whether or not a progress bar is printed to stdout.
    """
    return flatten1(
        http_get_all_pages(
            f"https://gitlab-master.nvidia.com/api/v4/projects/{project_id}/pipelines/{pipeline_id}/bridges",
            headers=headers,
            verbose=verbose,
        )
    )


def api_stages_for_pipeline(
    headers: GitLabHeaders,
    project_id: str,
    pipeline_id: str,
    *,
    verbose: ProgressBarLog = False,
) -> List[JSON]:
    """Gets all CI stages for a specific project's CI pipeline.

    HTTP GET /projects/:id/pipelines/:pipeline_id/jobs in the Gitalb API docs.

    :param:`verbose` controls whether or not a progress bar is printed to stdout.
    """
    return flatten1(
        http_get_all_pages(
            f"https://gitlab-master.nvidia.com/api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs",
            headers=headers,
            verbose=verbose,
        )
    )


def api_pipeline_details(
    headers: GitLabHeaders,
    project_id: str,
    pipeline_id: str,
) -> JSON:
    """Gets all details on a specific CI pipeline.

    HTTP GET /projects/:id/pipelines/:pipeline_id in the Gitalb API docs.

    :param:`verbose` controls whether or not a progress bar is printed to stdout.
    """
    return requests.get(
        f"https://gitlab-master.nvidia.com/api/v4/projects/{project_id}/pipelines/{pipeline_id}",
        headers=headers,
        params={},
    ).json()


#
#
# Functions calling multiple Gitlab APIs for a single purpose.
#
#


def http_get_bridge_trigger_for_pipelines(
    headers: GitLabHeaders,
    n_threads: Optional[int],
    project_id: str,
    pipeline_ids: Sequence[str],
    *,
    insert_pipeline_id: bool = True,
    verbose: bool = False,
) -> Tuple[List[JSON], List[str]]:
    """HTTP get all CI bridges from all of the supplied pipelines.

    A CI pipeline may have a stage that triggers the execute of another project's CI pipeline.
    These trigger stages are called bridges in the Gitlab API. This function obtains all
    of the bridged CI pipelines triggered stages using :func:`api_bridges_for_pipeline`.
    If you want to get the actual stages of the :param:`pipeline_ids`, use :func:`http_get_stages_for_pipelines`.

    The :param:`pipeline_ids` must correspond to valid CI pipelines.

    This function executes HTTP GET calls concurrenly: :param:`n_threads` controls the size of the backing
    threadpool for execution. See :func:`bionemo.ci.utils.execute` for details.

    By default, the pipeline ID from :param:`pipeline_ids` is associated with each obtained bridge CI stage.
    This is a modification of the original contents returned by the API.
    To disable, set :param:`insert_pipeline_id` to `False`.

    :param:`verbose` controls whether or not a progress bar is printed to stdout.

    NOTE: The CI pipeline must have exactly one bridge!
    """
    bridges = execute(
        [partial(api_bridges_for_pipeline, headers, project_id, pid, verbose=False) for pid in pipeline_ids],
        n_threads,
        verbose={"desc": "Getting CI bridge stages"} if verbose else False,
        throw_first_exception=True,
    )

    bridge_stages_when_failed = []
    pipeline_ids_without_trigger = []
    for fid, fid_bridge_stage in zip(pipeline_ids, bridges):
        if len(fid_bridge_stage) == 0:
            pipeline_ids_without_trigger.append(fid)
            continue
        assert (
            len(fid_bridge_stage) == 1
        ), f"pipeline {fid} had more than one bridge! ({len(fid_bridge_stage)}) {fid_bridge_stage=}"
        bridge_stage = fid_bridge_stage[0]
        if insert_pipeline_id:
            assert (
                fid == bridge_stage['pipeline']['id']
            ), f"Source pipeline id: {fid} != {bridge_stage['pipeline']['id']=}"
            bridge_stage['pipeline_id'] = bridge_stage['pipeline']['id']
        bridge_stages_when_failed.append(bridge_stage)
    return bridge_stages_when_failed, pipeline_ids_without_trigger


def http_get_stages_for_pipelines(
    headers: GitLabHeaders,
    n_threads: Optional[int],
    project_id: str,
    pipeline_ids: Sequence[str],
    *,
    insert_pipeline_id: bool = True,
    verbose: bool = False,
) -> List[JSON]:
    """HTTP GET all CI stages for the supplied pipelines.

    These are the CI stages that are defined for the pipeline IDs (:param:`pipeline_ids`). Internally, it uses
    :func:`api_stages_for_pipeline`. Explicitly, these returned stages do not include any CI stages from bridged pipelines.
    It will only contain the trigger stages for bridges. Use :func:`http_get_bridge_trigger_for_pipelines` to get these.

    The :param:`pipeline_ids` must correspond to valid CI pipelines.

    This function executes HTTP GET calls concurrenly: :param:`n_threads` controls the size of the backing
    threadpool for execution. See :func:`bionemo.ci.utils.execute` for details.

    By default, the pipeline ID from :param:`pipeline_ids` is associated with each obtained bridge CI stage.
    This is a modification of the original contents returned by the API.
    To disable, set :param:`insert_pipeline_id` to `False`.

    :param:`verbose` controls whether or not a progress bar is printed to stdout.
    """
    jobs: List[JSON] = []
    for jobs_for_pipeline in execute(
        [partial(api_stages_for_pipeline, headers, project_id, pid, verbose=False) for pid in pipeline_ids],
        n_threads,
        verbose={"desc": "Getting CI job stages for pipelines"} if verbose else False,
        throw_first_exception=True,
    ):
        if insert_pipeline_id:
            for j in jobs_for_pipeline:
                j['pipeline_id'] = j['pipeline']['id']
        jobs.extend(jobs_for_pipeline)
    return jobs


def http_get_details_for_pipelines(
    headers: GitLabHeaders,
    n_threads: Optional[int],
    project_id: str,
    pipeline_ids: Sequence[str],
    *,
    rename_pipeline_id: bool = True,
    verbose: bool = False,
) -> List[JSON]:
    """HTTP GET detailed information for each supplied pipeline ID.

    Uses the Gitlab API to obtain details on all supplied pipeline IDs (:param:`pipeline_ids`). Internally, it uses
    :func:`api_pipeline_details`.

    The :param:`pipeline_ids` must correspond to valid CI pipelines.

    This function executes HTTP GET calls concurrenly: :param:`n_threads` controls the size of the backing
    threadpool for execution. See :func:`bionemo.ci.utils.execute` for details.

    If :param:`rename_pipeline_id` is true, then the `'id'` field is renamed to `"pipeline_id"`.
    Otherwise, it is left as-is. By default, it is renamed.

    :param:`verbose` controls whether or not a progress bar is printed to stdout.
    """
    pipelines_details: List[JSON] = []
    for details_for_pipeline in execute(
        [partial(api_pipeline_details, headers, project_id, pid) for pid in pipeline_ids],
        n_threads,
        verbose={"desc": "Getting CI pipeline details"} if verbose else False,
        throw_first_exception=True,
    ):
        if rename_pipeline_id:
            details_for_pipeline['pipeline_id'] = details_for_pipeline['id']
            del details_for_pipeline['id']
        pipelines_details.append(details_for_pipeline)
    return pipelines_details
