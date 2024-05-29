status=0

jet_pipeline_id=$(curl --silent --header "PRIVATE-TOKEN: ${RO_API_TOKEN}" "https://gitlab-master.nvidia.com/api/v4/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/bridges" | jq '.[0].downstream_pipeline.id')
echo "JET_PIPELINE_ID=${jet_pipeline_id}"
python internal/jet/get_results_from_jet.py --pipeline_id $jet_pipeline_id  --save_dir . -vvv

# Extracts the 10th and 11th column from the csv file with information about jobs in jet pipeline corresponding
# to workload id and job key and creates workload_ids and job_keys arrays.
# Save the original IFS
filename="jet_query_${jet_pipeline_id}.json"
test_status=($(jq -r '.[].jet_test_status' "$filename"))
job_keys=($(jq -r '.[].job_key' "$filename"))

# Check if workload_ids is empty or if workload_ids and job_keys have different lengths
if [[ -z "${test_status[*]}" || ${#test_status[@]} -ne ${#job_keys[@]} ]]; then
  echo "Error: no jet tests run or jet_test and job_keys have different lengths."
  exit 1
fi

echo "==================================================================================================================="
echo "============================================ STARTING JET TEST ===================================================="
echo "==================================================================================================================="
for (( i=1; i<${#test_status[@]}; ++i)); do
  jet_test_status=${test_status[$i]}
  job_key=${job_keys[$i]}
  if [[ "${job_key}" == recipe/* && "${jet_test_status}" != "Success" ]]; then
      echo -e "\nJOB_KEY: ${job_key}, TEST: ${jet_test_status}"
      status+=1
  fi
done
echo "PIPELINE TEST STATUS: ${status}"
exit $status
