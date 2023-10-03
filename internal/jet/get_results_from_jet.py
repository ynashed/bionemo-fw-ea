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


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def get_results_from_jet(
    jet_workloads_ref: Optional[str],
    pipeline_id: Optional[int] = None,
    s_id: Optional[int] = None,
    jet_workloads_ref_pattern: Optional[str] = None,
    duration: Optional[str] = None,
    limit: Optional[int] = 1000,
    only_completed: bool = False,
    save_dir: Optional[str] = None,
    print_script: bool = False,
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
        jet_workloads_ref: optional, reference (branch) in JET Workloads Registry
        pipeline_id: optional, pipeline ID in JET CI
        s_id: optional, workload_id in JET CI
        jet_workloads_ref_pattern: optional, regex of query to filter results from Kibana, optional
        duration: optional, specifies period in the past to include jobs from till now,
                  either by date from which to include as ISO-8601-formatted datetime string (ie '2023-01-01T15:00' or '2023-01-01')
                  or duration, ie 1d, 5d, 1w, 2w, 1M, 2M, 1y, 2y
        limit: default set to 1000, limit of results to extract
        only_completed: default True, should only successfully completed jobs be returned?
        save_dir: optional, directory to save csv with results
        print_script: if to print script with commands to the console when the job execution failed and docker info
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

    if limit is not None:
        # FIXME: if filters are used, the query outputs only 10 top results so set high limit to get all results
        query = query.limit(limit)

    results = log_service.query(query)
    if len(results) == 0:
        logging.warning("No results found.")
        return

    logging.info(f'Getting {len(results)} jobs from Kibana... \n')
    output = []

    pipelines = []
    for result in tqdm.tqdm(results, desc="Loading data from Kibana"):
        output_i = {'duration': result.get("d_duration", None), 'timestamp': result.get('@timestamp', None)}

        ci_info = result.get("obj_ci", {})
        workloads_info = result.get("obj_workloads_registry", {})
        obj_workload = result.get("obj_workload", {})

        pipeline_id = ci_info.get("l_pipeline_id")
        if str(pipeline_id) not in pipelines:
            pipelines += [str(pipeline_id)]

        output_i["jet_workloads_ref"] = workloads_info.get("s_commit_ref", None)
        output_i["ci_pipeline_id"] = pipeline_id
        output_i["ci_job_id"] = ci_info.get("l_job_id")
        output_i["ci_job_duration"] = ci_info.get("d_job_duration", None)
        output_i["ci_job_name"] = ci_info.get("s_job_name")
        output_i["ci_job_status"] = ci_info.get("s_job_status")

        output_i["user"] = result["s_user"]
        output_i["s_id"] = result.get("s_id", None)
        output_i["job_key"] = obj_workload.get("s_key", None)
        output_i["job_type"] = obj_workload.get("s_type", None)
        output_i["exit_code"] = (
            result.get("l_exit_code", None) if output_i["job_type"] == "recipe" else int(result.get("b_invalid", 1))
        )

        msg_output_details = (
            f'{output_i["job_key"]} with status: '
            f'{"SUCCESS" if (output_i["exit_code"] is not None and output_i["exit_code"] == 0) else "FAILED"} '
            f'\nJET Workloads ref: {output_i["jet_workloads_ref"]}, JET pipeline id: {pipeline_id}, '
            f'JET job id: {output_i["ci_job_id"]}, JET workload id: {output_i["s_id"]}, '
            f'Timestamp: {output_i["timestamp"]}, '
        )

        if output_i['ci_job_duration'] is not None:
            msg_output_details += f'\nJET job duration: {round(output_i["ci_job_duration"], 3)}s'

        if output_i['duration'] is not None:
            msg_output_details += f"\nScript execution time: {round(output_i['duration'], 3)}s"

        if output_i["job_type"] == "build":
            docker_img_info = obj_workload["obj_spec"]["obj_source"]
            msg = f"DOCKER BUILD {msg_output_details}"

            for k, v in docker_img_info.items():
                name = f"docker_{k}"
                output_i[name] = v
                if save_dir is None and print_script:
                    msg += f"\nDocker info:\n{k}: {v}\n\n"

        elif output_i["job_type"] == "recipe":
            output_i["script"] = obj_workload['obj_spec']['s_script']

            if "conv" in output_i["job_key"]:
                prefix = "CONVERGENCE TEST"
            elif "perf" in output_i["job_key"]:
                prefix = "PERFORMANCE TEST"
            else:
                prefix = "TEST"

            msg = f'{prefix} {msg_output_details}'

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

                output_i[name] = asset["s_url"]
                if save_dir is None:
                    msg += f'\n{name}: {output_i[name]}'

            if save_dir is None and output_i["exit_code"] != 0 and print_script:
                msg += f"\n\nScript:\n {output_i['script']}\n\n"

            if save_dir is not None:
                output_i["obj_spec"] = obj_workload.get("obj_spec", {})
                output_i["obj_workloads_registry"] = workloads_info

                output_i["env_info"] = {}
                output_i["env_info"]["gpu"] = result.get("nested_gpu", None)
                output_i["env_info"]["cpu"] = result.get("obj_cpu", None)
        else:
            raise ValueError("Only job_type recipe or build are supported.")

        logging.info(msg + "\n\n")
        output.append(output_i)

    df = pd.DataFrame(output)
    if s_id is None and len(results) > 1:
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
        df.to_csv(filepath)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--jet_workloads_ref', type=str, default=None, help='Reference (branch) in JET Workloads Registry, optional'
    )
    parser.add_argument('--pipeline_id', type=int, default=None, help='Pipeline ID in JET CI, optional')
    parser.add_argument('--s_id', type=str, default=None, help='Workload ID in JET CI, optional')
    parser.add_argument('--d', type=str, default=None, help='specifies period in the past to include jobs from')
    parser.add_argument(
        '--jet_workloads_ref_pattern',
        type=str,
        default=None,
        help='Regex of query to filter results from Kibana, optional',
    )
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save csv with results, optional')
    parser.add_argument('--only_completed', action='store_true', help='Should only completed pipelines be returned?')
    parser.add_argument(
        '--print_script',
        action='store_true',
        help='Should commands of failed tests be printed to the console as well as docker info?',
    )
    parser.add_argument('--limit', type=int, default=1000, help='Limit number of printed results')

    args = parser.parse_args()
    get_results_from_jet(
        jet_workloads_ref=args.jet_workloads_ref,
        pipeline_id=args.pipeline_id,
        s_id=args.s_id,
        jet_workloads_ref_pattern=args.jet_workloads_ref_pattern,
        duration=args.d,
        save_dir=args.save_dir,
        only_completed=args.only_completed,
        limit=args.limit,
        print_script=args.print_script,
    )
