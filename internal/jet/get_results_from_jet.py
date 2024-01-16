import datetime
import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
import tqdm
from jet.logs.queries import Field, JETLogsQuery
from jet.utils.instance import JETInstance
from tabulate import tabulate


# using logging not from NeMo to run this script outside a container
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def filter_out_failed_and_rerun_jobs(df: pd.DataFrame, n_all_results: int) -> pd.DataFrame:
    """
    Filters out records of jobs which have been rerun, ie due to infrastructure failures.
    """
    df["tmp_label"] = df["jet_ci_pipeline_id"].astype(str) + df["job_key"]
    df.drop_duplicates('tmp_label', keep='last', inplace=True)
    df.drop(labels="tmp_label", axis=1, inplace=True)
    print("\n")
    logging.info(
        f'Keeping only the most recent jobs per pipeline id and job key: ' f'{df.shape[0]}/{n_all_results} jobs \n'
    )
    return df


def print_table_with_results(df: pd.DataFrame) -> None:
    """
    Print table with essential information about the jobs to the console table
    """
    table_dict = OrderedDict()
    table_dict["BioNeMo pipeline ID"] = df.bionemo_ci_pipeline_id
    table_dict["JET job ID"] = df.jet_ci_job_id
    table_dict["Job Type"] = df.job_type
    table_dict["Job Key"] = df.job_key.str.split("/", expand=True)[1]
    table_dict["Status"] = df.status
    table_dict["Log URL"] = df.log_output_script if "log_output_script" in df else ""
    table = tabulate(table_dict, headers=list(table_dict.keys()), tablefmt="psql")

    print(table)
    if not all(df.exit_code == 0):
        logging.error(
            "Some jobs failed to complete successfully. Detailed information about jobs is available by appending -vv\n"
        )
    else:
        logging.info("All jobs completed successfully!")


def get_duration_stats(
    pipelines_info: dict, job_durations: pd.Series, script_durations: pd.Series, digits: int = 1
) -> None:
    """
    Calculates duration statistics for pipelines(s) and included jobs. Differentiates between
    job duration (including queuing time) and script duration (excluding queuing time)
    """
    print('\n')
    N = len(job_durations)
    pipelines_duration = pd.Series([v["duration"] for _, v in pipelines_info.items()])
    logging.info(
        f"Duration information for {N} jobs in {len(pipelines_info)} pipeline(s): {','.join(pipelines_info.keys())}\n"
    )
    duration_info = OrderedDict({"pipeline": pipelines_duration, "job": job_durations, "scripts": script_durations})

    duration_stats = OrderedDict()
    duration_stats["Duration (s)"] = list(duration_info.keys())
    duration_stats["total"] = [round(v.sum(), digits) for k, v in duration_info.items()]
    duration_stats["mean"] = [round(v.mean(), digits) for k, v in duration_info.items()]
    duration_stats["median"] = [round(v.median(), digits) for k, v in duration_info.items()]
    duration_stats["min"] = [round(v.min(), digits) for k, v in duration_info.items()]
    duration_stats["max"] = [round(v.max(), digits) for k, v in duration_info.items()]
    table = tabulate(duration_stats, headers=list(duration_stats.keys()), tablefmt="psql")
    print(table)


def query_jet_jobs(
    jet_instance: JETInstance,
    jet_workloads_ref: Optional[str] = None,
    pipeline_id: Optional[int] = None,
    job_id: Optional[int] = None,
    pipeline_type: Optional[str] = None,
    duration: Optional[str] = None,
    limit: Optional[int] = 1000,
    only_completed: bool = False,
) -> List[dict]:
    """
    Queries Kibana to get results for JET jobs according to the specifications.
    """

    log_service = jet_instance.log_service()
    query = JETLogsQuery().filter(Field("obj_workload.obj_labels.origin") == "bionemo")

    if pipeline_type is not None:
        logging.info(f'Query results for pipeline_type: {pipeline_type}')
        query = query.filter(Field("obj_workload.obj_labels.workload_ref") == f"bionemo/{pipeline_type}")

    if jet_workloads_ref is not None:
        logging.info(f'Query results for jet_workloads_ref: {jet_workloads_ref}')
        query = query.filter(Field("obj_workloads_registry.s_commit_ref") == jet_workloads_ref)

    if pipeline_id is not None:
        logging.info(f'Query results for Jet CI pipeline id: {pipeline_id}')
        query = query.filter(Field("obj_ci.l_pipeline_id") == pipeline_id)

    if job_id is not None:
        logging.info(f'Query results for Jet CI job id: {job_id}')
        query = query.filter(Field("obj_ci.l_job_id") == job_id)

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

    # TODO(dorotat): if filters are used, the query outputs only 10 top results so set high limit to get all results
    if limit is not None:
        query = query.limit(limit)

    # Getting results for all jobs in the query
    jobs_results = log_service.query(query)
    return jobs_results


def log_detailed_job_info(df: pd.DataFrame) -> None:
    """
    Logs to the console detailed summary of JET test execution such as job id and the status of the job,
    its duration as well as script to execute if the job failed
    """

    for _, job_info in df.iterrows():
        msg = (
            f'{job_info["job_key"]} with status: {job_info["status"]} '
            f'\nJET pipeline id: {job_info["jet_ci_pipeline_id"]}, BioNeMo pipeline id: {job_info["bionemo_ci_pipeline_id"]} '
            f'JET job id: {job_info["jet_ci_job_id"]}, '
            f'job duration: {job_info["jet_ci_job_duration"]}, script duration: {job_info["script_duration"]}'
        )

        if job_info["job_type"] == "build":
            msg = f"DOCKER BUILD {msg}"
            for k, v in job_info[job_info.index.str.startswith('docker_')].items():
                msg += f"\nDocker info: \n{k}: {v}\n"

        elif job_info["job_type"] == "recipe":
            if "conv" in job_info["job_key"]:
                prefix = "CONVERGENCE TEST"
            elif "perf" in job_info["job_key"]:
                prefix = "PERFORMANCE TEST"
            else:
                prefix = "TEST"
            msg = f'{prefix} {msg}'

            for k, v in job_info[job_info.index.str.startswith('log_')].items():
                msg += f'\n{k.replace("log_", "")}: {v}'

            if job_info["exit_code"] != 0:
                msg += f"\n\nScript:\n {job_info['script']}\n\n"
        logging.info(msg + '\n')


def get_docker_info(docker_info: dict) -> dict:
    """
    Parse information about the docker container from JET jobs results
    """
    new_docker_info = {}
    for k, v in docker_info.items():
        name = f"docker_{k}"
        new_docker_info[name] = v
    return new_docker_info


def get_job_logs_info(job_logs: dict) -> dict:
    """
    Parse information about available logs per JET job
    """
    new_job_info = {}
    for asset in job_logs:
        if asset["s_name"] == 'dllogger.json':
            name = "log_dllogger"
        elif asset["s_name"] == 'output_script-0.log':
            name = "log_output_script"
        elif asset["s_name"] == 'trainer_logs.json':
            name = "log_trainer_logs"
        elif asset["s_name"] == "error_msg.txt":
            name = "log_error_msg"
        else:
            continue
        new_job_info[name] = asset["s_url"]
    return new_job_info


def get_job_info(job_raw_result: dict, save_dir: Optional[str]) -> dict:
    """
    Parse information about JET job from results obtained from JET
    """
    job_info = {
        'script_duration': job_raw_result.get("d_duration", None),
        'timestamp': job_raw_result.get('@timestamp', None),
    }

    ci_info = job_raw_result.get("obj_ci", {})
    workloads_info = job_raw_result.get("obj_workloads_registry", {})
    obj_workload = job_raw_result.get("obj_workload", {})

    pipeline_id = str(ci_info.get("l_pipeline_id"))

    job_info["jet_workloads_ref"] = workloads_info.get("s_commit_ref", None)
    job_info["jet_ci_pipeline_id"] = pipeline_id
    job_info["jet_ci_job_id"] = ci_info.get("l_job_id")
    job_info["jet_ci_job_duration"] = ci_info.get("d_job_duration", None)
    job_info["jet_ci_job_name"] = ci_info.get("s_job_name")
    job_info["jet_ci_job_status"] = ci_info.get("s_job_status")
    # Workload id is needed for JET test
    job_info["jet_ci_workload_id"] = job_raw_result.get("s_id", None)

    job_info["user"] = job_raw_result["s_user"]
    job_info["job_key"] = obj_workload.get("s_key", None)
    job_info["job_type"] = obj_workload.get("s_type", None)
    job_info["exit_code"] = (
        job_raw_result.get("l_exit_code", None)
        if job_info["job_type"] == "recipe"
        else int(job_raw_result.get("b_invalid", 1))
    )
    job_info["status"] = "SUCCESS" if (job_info["exit_code"] is not None and job_info["exit_code"] == 0) else "FAILED"
    if "recipe" in job_info["job_key"]:
        job_info["script"] = obj_workload['obj_spec'].get('s_script', None)

    if job_info["job_type"] == "build":
        docker_info = get_docker_info(docker_info=obj_workload["obj_spec"]["obj_source"])
        job_info.update(docker_info)

    elif job_info["job_type"] == "recipe":
        jobs_logs_info = get_job_logs_info(job_logs=job_raw_result.get("nested_assets", []))
        job_info.update(jobs_logs_info)

        if save_dir is not None:
            job_info["env_info_gpu"] = job_raw_result.get("nested_gpu", None)
            job_info["env_info_cpu"] = job_raw_result.get("obj_cpu", None)
    else:
        raise ValueError("Only jobs of type 'recipe' or 'build' are supported.")

    return job_info


def get_results_from_jet(
    jet_workloads_ref: Optional[str] = None,
    pipeline_id: Optional[int] = None,
    job_id: Optional[int] = None,
    pipeline_type: Optional[str] = None,
    duration: Optional[str] = None,
    limit: Optional[int] = 1000,
    only_completed: bool = False,
    save_dir: Optional[str] = None,
    all_jobs: bool = False,
    verbosity_level: int = 0,
    return_df: bool = False,
) -> Optional[pd.DataFrame]:
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

    # Attach to your command --pipeline_type dev to get logs for all JET test pipelines which were execute for branch "dev"
    python internal/jet/get_results_from_jet.py ..... --pipeline_type dev

    # Or attach to your command --save_dir <FOLDER_PATH> to save details of the run tests to a csv with predefined path
    python internal/jet/get_results_from_jet.py ..... --save_dir <FOLDER_PATH>

    # Attach to your command -v or -vv to have more detailed levels of verbosity of logs
    python internal/jet/get_results_from_jet.py ..... -vv

    Args:
        jet_workloads_ref: a reference (branch) in JET Workloads Registry, optional
        pipeline_id: a pipeline id in JET CI, optional
        job_id: a job id in JET CI, optional,
        pipeline_type: specifies type of BioNeMo's CI pipeline(s) to get jet logs for, optional.
                       Can be either "merge_request_event" or name of the git branch in BioNeMo repo, ie "dev"
        duration: specifies period in the past to include jobs from up to now, optional. The accepted formats are
                  either dates from which to include as ISO-8601-formatted datetime string (ie '2023-01-01T15:00','2023-01-01')
                  or durations, ie 1d, 5d, 1w, 2w, 1M, 2M, 1y, 2y
        limit: a limit of results to extract, default set to 1000,
        only_completed: a flag that determines whether to keep only successfully completed jobs (exit status != 0)
        save_dir: directory to save csv with results, optional
        all_jobs: a flag that determines whether to take all jobs in a pipeline or keep only the most recent jobs for each job key.
                     This is useful when individual jobs fail due to infrastructure issues and are rerun within
                     the same JET pipeline.
        verbosity_level: a verbosity level of logs
        return_df: a flag that determines whether to return pd.DataFrame with job details
    """
    jet_instance = JETInstance(env="prod")
    results_jobs = query_jet_jobs(
        jet_instance=jet_instance,
        jet_workloads_ref=jet_workloads_ref,
        pipeline_id=pipeline_id,
        job_id=job_id,
        pipeline_type=pipeline_type,
        duration=duration,
        limit=limit,
        only_completed=only_completed,
    )

    if len(results_jobs) == 0:
        logging.warning("No results found.")
        return

    logging.info(f'Getting {len(results_jobs)} jobs from Kibana... \n')
    output = []

    jet_ci = jet_instance.gitlab_ci()
    pipelines_info = {}
    for result in tqdm.tqdm(results_jobs, desc="Getting results for JET tests "):
        job_info = get_job_info(job_raw_result=result, save_dir=save_dir)

        pipeline_id = job_info["jet_ci_pipeline_id"]
        if job_info["jet_ci_pipeline_id"] not in pipelines_info:
            obj_workload = result.get("obj_workload", {})
            pipelines_info[pipeline_id] = obj_workload["obj_labels"]
            pipelines_info[pipeline_id]["duration"] = jet_ci.project.pipelines.get(pipeline_id).duration

        job_info["jet_ci_pipeline_duration"] = pipelines_info[pipeline_id]["duration"]
        job_info["bionemo_ci_pipeline_id"] = pipelines_info[pipeline_id].get("bionemo_ci_pipeline_id", None)
        job_info["pipeline_type"] = pipelines_info[pipeline_id].get("workload_ref", None)

        output.append(job_info)

    df = pd.DataFrame(output)
    df.sort_values(by=["jet_ci_pipeline_id", "job_key", "timestamp"], inplace=True)

    # Keeping only the most recent jobs per pipeline and job_key (to filter out jobs that were rerun)
    if not all_jobs:
        df = filter_out_failed_and_rerun_jobs(df=df, n_all_results=df.shape[0])

    if verbosity_level > 1:
        log_detailed_job_info(df=df)

    if verbosity_level >= 1:
        print_table_with_results(df=df)

    # Calculating duration-related analytics (overall, pipeline and job specific)
    if verbosity_level > 0 and df.shape[0] > 2:
        get_duration_stats(
            pipelines_info=pipelines_info, job_durations=df.jet_ci_job_duration, script_durations=df.script_duration
        )

    if save_dir is not None:
        filename = f'jet_query_{"_".join(pipelines_info.keys())}.csv'
        filepath = os.path.join(save_dir, filename)
        logging.info(f"Saving query results to: {filepath}")
        df.to_csv(filepath, index=False)
    if return_df:
        return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--jet_workloads_ref', type=str, default=None, help='Reference (branch) in JET Workloads Registry, optional'
    )
    parser.add_argument('--pipeline_id', type=int, default=None, help='Pipeline ID in JET CI, optional')
    parser.add_argument('--job_id', type=int, default=None, help='Job ID in JET CI, optional')
    parser.add_argument('--d', type=str, default=None, help='Specifies period in the past to include jobs from')
    parser.add_argument(
        '--pipeline_type',
        type=str,
        default=None,
        help="Specifies type of BioNeMo's CI pipeline(s) to get jet logs for, optional. Can be either 'merge_request' "
        "or name of the git branch in BioNeMo repo, ie 'dev'",
    )
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save csv with results, optional')
    parser.add_argument(
        '--return_df',
        action='store_true',
        help='A flag that determines whether to return pd.DataFrame with job results',
    )
    parser.add_argument(
        '--only_completed',
        action='store_true',
        help='A flag that determines whether to keep only ' 'successfully completed jobs (exit status != 0)',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='A flag that determines whether to take all jobs in the pipeline or keep only the most recent jobs for each job key. '
        'Useful when some jobs have been rerun due to infrastructure issues.',
    )
    parser.add_argument('--limit', type=int, default=1000, help='Limit number of printed results')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='a flag that determines verbosity level')

    args = parser.parse_args()
    get_results_from_jet(
        jet_workloads_ref=args.jet_workloads_ref,
        pipeline_id=args.pipeline_id,
        job_id=args.job_id,
        pipeline_type=args.pipeline_type,
        duration=args.d,
        save_dir=args.save_dir,
        only_completed=args.only_completed,
        limit=args.limit,
        all_jobs=args.all,
        verbosity_level=args.verbose,
        return_df=args.return_df,
    )
