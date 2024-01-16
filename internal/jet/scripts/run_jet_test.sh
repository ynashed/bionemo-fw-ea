# For internal use only. Should not be run manually. It is employed in CI.
status=0

jet_pipeline_id=$(curl --silent --header "PRIVATE-TOKEN: ${RO_API_TOKEN}" "https://gitlab-master.nvidia.com/api/v4/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/bridges" | jq '.[0].downstream_pipeline.id')
python internal/jet/get_results_from_jet.py --pipeline_id $jet_pipeline_id  --save_dir . -vv

# Extracts the 10th and 11th column from the csv file with information about jobs in jet pipeline corresponding
# to workload id and job key and creates workload_ids and job_keys arrays.
filename="jet_query_${jet_pipeline_id}.csv"
workload_ids=($(cat $filename | cut -d ',' -f9))
echo $workload_ids
job_keys=($(cat $filename  | cut -d ',' -f11))


if [[ -z "${workload_ids}" ]]
then
  exit 1
fi

echo "==================================================================================================================="
echo "============================================ STARTING JET TEST ===================================================="
echo "==================================================================================================================="


for (( i=1; i<${#workload_ids[@]}; ++i)); do
  jet_log_id=${workload_ids[$i]}
  job_key=${job_keys[$i]}
  if [[ "${job_key}" == recipe/* ]]
  then
    echo "\nJOB_KEY: ${job_key}, WORKLOAD ID: ${jet_log_id}"
    jet tests run static --check-status --config "exit_codes[0]=0" --workload-log-id "${jet_log_id}" --origin "bionemo" --user "${GITLAB_USER_LOGIN}" --output "test-${jet_log_id}.zip"
    job_status=$?
    status=$(($job_status > $status ? $job_status : $status))
    jet logs upload "test-${jet_log_id}.zip"
    echo "JOB TEST STATUS: ${job_status}, PIPELINE TEST STATUS: ${status}"
  fi
done

exit $status