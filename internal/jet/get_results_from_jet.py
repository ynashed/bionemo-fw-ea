import datetime
import logging
import os
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import pandas as pd
import tqdm
from jet.logs.queries import Field, JETLogsQuery
from jet.utils.instance import JETInstance


# using logging not from NeMo to run this script outside a container
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def get_results_from_jet(
    jet_workloads_ref: Optional[str],
    pipeline_id: Optional[int] = None,
    job_id: Optional[int] = None,
    s_id: Optional[int] = None,
    jet_workloads_ref_pattern: Optional[str] = None,
    duration: Optional[str] = None,
    limit: Optional[int] = 1000,
    only_completed: bool = False,
    save_dir: Optional[str] = None,
    print_script: bool = False,
    most_recent: bool = False,
) -> None:
    """
    Queries and outputs results (print or saved as csv) of tests run in JET given reference (branch)
    from JET Workloads Registry, its pattern, or related pipeline id in JET CI.
    If no query arguments are provided, all results run on bionemo account are returned.

    By default, the basic info about the tests, their logs and related jobs in JET is printed to the console.
    If save_dir is provided, more detailed information is saved as csv

    Requires installed JET API https://gitlab-master.nvidia.com/dl/jet/api. Can be executed outside docker container.

    Usage:

    # Get results for all tests run for BioNeMo account (main tests and the ones run on ephemeral/bionemo)
    python internal/jet/get_results_from_jet.py

    # Get reference-related results
    python internal/jet/get_results_from_jet.py --jet_workloads_ref bionemo/training-unit-tests

    # Get pipeline id-related results
    python internal/jet/get_results_from_jet.py --pipeline_id 9469458

    # Get results for tests completed on dev branch for
    python internal/jet/get_results_from_jet.py --jet_workloads_ref_pattern "ephemeral/bionemo/dev/.*"

    # Get results for tests based on reference pattern, ie ephemeral branches
    python internal/jet/get_results_from_jet.py --jet_workloads_ref_pattern "ephemeral/bionemo/.*"

    # Attach to your command --print_script to print commands of the failed tests and related docker info to the console
    python internal/jet/get_results_from_jet.py ..... --print_script

    # Or attach to your command --save_dir <FOLDER_PATH> to save details of the run tests to a csv with predefined path
    python internal/jet/get_results_from_jet.py ..... --save_dir <FOLDER_PATH>

    Args:
        jet_workloads_ref: a reference (branch) in JET Workloads Registry, optional
        pipeline_id: a pipeline id in JET CI, optional
        job_id: a job id in JET CI, optional,
        s_id: a workload_id in JET CI, optional,
        jet_workloads_ref_pattern: a regex of query to filter results from Kibana, optional
        duration: specifies period in the past to include jobs from up to now, optional. The accepted formats are
                  either dates from which to include as ISO-8601-formatted datetime string (ie '2023-01-01T15:00','2023-01-01')
                  or durations, ie 1d, 5d, 1w, 2w, 1M, 2M, 1y, 2y
        limit: a limit of results to extract, default set to 1000,
        only_completed: a flag that determines whether to keep only successfully completed jobs (exit status != 0)
        save_dir: directory to save csv with results, optional
        print_script: a flag that determines whether to print commands and docker information to the console for
                      JET jobs which failed (exit status != 0)
        most_recent: a flag that determines whether to keep only the most recent jobs for each job key.
                     This is useful when individual jobs fail due to infrastructure issues and are rerun within
                     the same JET pipeline.
    """

    jet_instance = JETInstance(env="prod")
    log_service = jet_instance.log_service()
    query = JETLogsQuery().filter(Field("obj_workload.obj_labels.origin") == "bionemo")

    if jet_workloads_ref_pattern is not None:
        logging.info(f'Query results for jet_workloads_ref pattern: {jet_workloads_ref_pattern}')
        query = query.filter(Field("obj_workloads_registry.s_commit_ref").matches(jet_workloads_ref_pattern))

    if jet_workloads_ref is not None:
        logging.info(f'Query results for jet_workloads_ref: {jet_workloads_ref}')
        query = query.filter(Field("obj_workloads_registry.s_commit_ref") == jet_workloads_ref)

    if pipeline_id is not None:
        logging.info(f'Query results for Jet CI pipeline id: {pipeline_id}')
        query = query.filter(Field("obj_ci.l_pipeline_id") == pipeline_id)

    if job_id is not None:
        logging.info(f'Query results for Jet CI job id: {job_id}')
        query = query.filter(Field("obj_ci.l_job_id") == job_id)

    if s_id is not None:
        logging.info(f'Query results for Jet CI workload id: {s_id}')
        query = query.filter(Field("s_id") == s_id)

    if duration is not None:
        if any(duration.endswith(s) for s in ['d', 'w', 'M', 'y']):
            query = query.filter(Field('ts_created') >= f'now-{duration}')
        else:
            try:
                datetime.date.fromisoformat(duration)
                query = query.filter(Field('ts_created') >= duration)

            except ValueError:
                logging.error(f"Invalid duration string: {str}. Proceeding without filtering results by date")

    if only_completed:
        query = query.filter(Field("l_exit_code") == 0.0)

    # TODO(@dorotat): if filters are used, the query outputs only 10 top results so set high limit to get all results
    if limit is not None:
        query = query.limit(limit)

    # Getting results for all jobs in the query
    results_per_job = log_service.query(query)
    if len(results_per_job) == 0:
        logging.warning("No results found.")
        return

    logging.info(f'Getting {len(results_per_job)} jobs from Kibana... \n')
    output = []

    pipelines = []
    for result in tqdm.tqdm(results_per_job, desc="Loading data from Kibana"):
        info_job = {'duration': result.get("d_duration", None), 'timestamp': result.get('@timestamp', None)}

        ci_info = result.get("obj_ci", {})
        workloads_info = result.get("obj_workloads_registry", {})
        obj_workload = result.get("obj_workload", {})

        pipeline_id = ci_info.get("l_pipeline_id")
        if str(pipeline_id) not in pipelines:
            pipelines += [str(pipeline_id)]

        info_job["jet_workloads_ref"] = workloads_info.get("s_commit_ref", None)
        info_job["ci_pipeline_id"] = pipeline_id
        info_job["ci_job_id"] = ci_info.get("l_job_id")
        info_job["ci_job_duration"] = ci_info.get("d_job_duration", None)
        info_job["ci_job_name"] = ci_info.get("s_job_name")
        info_job["ci_job_status"] = ci_info.get("s_job_status")

        info_job["user"] = result["s_user"]
        info_job["s_id"] = result.get("s_id", None)
        info_job["job_key"] = obj_workload.get("s_key", None)
        info_job["job_type"] = obj_workload.get("s_type", None)
        info_job["exit_code"] = (
            result.get("l_exit_code", None) if info_job["job_type"] == "recipe" else int(result.get("b_invalid", 1))
        )

        msg_output_details = (
            f'{info_job["job_key"]} with status: '
            f'{"SUCCESS" if (info_job["exit_code"] is not None and info_job["exit_code"] == 0) else "FAILED"} '
            f'\nJET Workloads ref: {info_job["jet_workloads_ref"]}, JET pipeline id: {pipeline_id}, '
            f'JET job id: {info_job["ci_job_id"]}, JET workload id: {info_job["s_id"]}, '
            f'Timestamp: {info_job["timestamp"]}, '
        )

        if info_job['ci_job_duration'] is not None:
            msg_output_details += f'\nJET job duration: {round(info_job["ci_job_duration"], 3)}s'

        if info_job['duration'] is not None:
            msg_output_details += f"\nScript execution time: {round(info_job['duration'], 3)}s"

        if info_job["job_type"] == "build":
            docker_img_info = obj_workload["obj_spec"]["obj_source"]
            msg = f"DOCKER BUILD {msg_output_details}"

            for k, v in docker_img_info.items():
                name = f"docker_{k}"
                info_job[name] = v
                if save_dir is None and print_script:
                    msg += f"\nDocker info:\n{k}: {v}\n\n"

        elif info_job["job_type"] == "recipe":
            if "conv" in info_job["job_key"]:
                prefix = "CONVERGENCE TEST"
            elif "perf" in info_job["job_key"]:
                prefix = "PERFORMANCE TEST"
            else:
                prefix = "TEST"

            msg = f'{prefix} {msg_output_details}'

            # Saving paths to the job-related logs
            for asset in result.get("nested_assets", []):
                if asset["s_name"] == 'dllogger.json':
                    name = "dllogger"
                elif asset["s_name"] == 'output_script.log':
                    name = "output_script"
                elif asset["s_name"] == 'trainer_logs.json':
                    name = "trainer_logs"
                elif asset["s_name"] == "error_msg.txt":
                    name = "error_msg"
                else:
                    continue

                info_job[name] = asset["s_url"]
                msg += f'\n{name}: {info_job[name]}'

            if info_job["exit_code"] != 0 and print_script:
                msg += f"\n\nScript:\n {obj_workload['obj_spec']['s_script']}\n\n"

            if save_dir is not None:
                info_job["env_info_gpu"] = result.get("nested_gpu", None)
                info_job["env_info_cpu"] = result.get("obj_cpu", None)
        else:
            raise ValueError("Only job_type recipe or build are supported.")

        logging.info(msg + "\n\n")
        output.append(info_job)

    df = pd.DataFrame(output)
    df.sort_values(by=["ci_pipeline_id", "job_key", "timestamp"], inplace=True)

    # Keeping only the most recent jobs per pipeline and job_key (to filter out jobs that were rerun)
    if most_recent:
        df["tmp_label"] = df["ci_pipeline_id"].astype(str) + df["job_key"]
        df.drop_duplicates('tmp_label', keep='last', inplace=True)
        df.drop(labels="tmp_label", axis=1, inplace=True)
        print("\n")
        logging.info(
            f'Keeping only the most recent jobs per ci_pipeline_id and job_key: '
            f'{df.shape[0]}/{len(results_per_job)} jobs \n'
        )

    # Calculating duration-related analytics (overall, pipeline and job specific)
    if s_id is None and len(results_per_job) > 1:
        print('\n\n')
        jet_ci = jet_instance.gitlab_ci()
        pipelines_duration = [jet_ci.project.pipelines.get(pipeline_id).duration for pipeline_id in pipelines]
        logging.info(
            f"Duration stats of jobs in {len(pipelines)} pipeline(s) {','.join(pipelines)} "
            f"resulting in {df.shape[0] - 1} jobs: \n"
            f"Pipeline completion: total {round(sum(pipelines_duration), 1)}s , mean: {round(np.mean(pipelines_duration), 1)}s, "
            f"median: {round(np.median(pipelines_duration), 1)}s, min: {round(min(pipelines_duration), 1)}s, max: {round(max(pipelines_duration), 1)}s\n"
            f"Jobs execution (scripts execution plus queueing time): : total: {round(df.ci_job_duration.sum(), 1)}s , mean: {round(df.ci_job_duration.mean(), 1)}s, "
            f"median: {round(df.ci_job_duration.median(), 1)}s, min: {round(df.ci_job_duration.min(), 1)}s, max: {round(df.ci_job_duration.max(), 1)}s\n"
            f"Scripts execution: total: {round(df.duration.sum(), 1)}s , mean: {round(df.duration.mean(), 1)}s, "
            f"median: {round(df.duration.median(), 1)}s, min: {round(df.duration.min(), 1)}s, max: {round(df.duration.max(), 1)}s"
        )
    if save_dir is not None:
        filename = f'jet_query_{"_".join(pipelines)}.csv'
        filepath = os.path.join(save_dir, filename)
        logging.info(f"Saving query results to: {filepath}")
        df.to_csv(filepath, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--jet_workloads_ref', type=str, default=None, help='Reference (branch) in JET Workloads Registry, optional'
    )
    parser.add_argument('--pipeline_id', type=int, default=None, help='Pipeline ID in JET CI, optional')
    parser.add_argument('--job_id', type=int, default=None, help='Job ID in JET CI, optional')
    parser.add_argument('--s_id', type=str, default=None, help='Workload ID in JET CI, optional')
    parser.add_argument('--d', type=str, default=None, help='specifies period in the past to include jobs from')
    parser.add_argument(
        '--jet_workloads_ref_pattern',
        type=str,
        default=None,
        help='Regex of query to filter results from Kibana, optional',
    )
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save csv with results, optional')
    parser.add_argument(
        '--only_completed',
        action='store_true',
        help='A flag that determines whether to keep only ' 'successfully completed jobs (exit status != 0)',
    )
    parser.add_argument(
        '--print_script',
        action='store_true',
        help='A flag that determines whether to print commands and docker information to the console for JET jobs '
        'which failed (exit status != 0)',
    )
    parser.add_argument(
        '--most_recent',
        action='store_true',
        help='A flag that determines whether to keep only the most recent jobs for each job key',
    )
    parser.add_argument('--limit', type=int, default=1000, help='Limit number of printed results')

    args = parser.parse_args()
    get_results_from_jet(
        jet_workloads_ref=args.jet_workloads_ref,
        pipeline_id=args.pipeline_id,
        job_id=args.job_id,
        s_id=args.s_id,
        jet_workloads_ref_pattern=args.jet_workloads_ref_pattern,
        duration=args.d,
        save_dir=args.save_dir,
        only_completed=args.only_completed,
        limit=args.limit,
        print_script=args.print_script,
        most_recent=args.most_recent,
    )
