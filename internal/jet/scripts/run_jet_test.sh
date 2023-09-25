# For internal use only. Should not be run manually. It is employed in CI.
status=0

jet_pipeline_id=$(curl --silent --header "PRIVATE-TOKEN: ${RO_API_TOKEN}" "https://gitlab-master.nvidia.com/api/v4/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/bridges" | jq '.[0].downstream_pipeline.id')
jet_logs_table=$(jet -c -tf plain -th logs query -c "obj_workload.s_key" -c s_id --eq obj_ci.l_pipeline_id "${jet_pipeline_id}" --re obj_workload.s_type "build|recipe")
if [[ -z "${jet_logs_table}" ]]
then
  exit 1
fi
for jet_log_id in ${jet_logs_table}; do
  if [[ $jet_log_id = recipe/* ]] || [[ $jet_log_id = build/* ]]
  then
    echo "==================================================================================================================="
    echo "======= JOB KEY: ${jet_log_id}"
    echo "==================================================================================================================="
    s_key=${jet_log_id}
  else
    echo "======= WORKLOAD ID: ${jet_log_id} ================================="
    python internal/jet/get_results_from_jet.py --pipeline_id ${jet_pipeline_id} --s_id ${jet_log_id} --print_script
    if [[ $s_key = recipe/* ]]
    then
      jet tests run static --check-status --config "exit_codes[0]=0" --workload-log-id "${jet_log_id}" --origin "bionemo" --user "${GITLAB_USER_LOGIN}" --output "test-${jet_log_id}.zip" || status=$(($? > $status ? $? : $status))
      jet logs upload "test-${jet_log_id}.zip"
      echo "status: ${status}"
    fi
  fi
done

exit $status