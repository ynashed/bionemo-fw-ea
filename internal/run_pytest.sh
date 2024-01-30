#!/bin/bash


test_files=($(find tests/ examples/tests -type f -name "test_*.py" ))
n_test_files=${#test_files[@]}
counter=1
# the overall test status collected from all pytest commands
status=0

for testfile in "${test_files[@]}"; do
  rm -rf ./.pytest_cache/
  set -x
  echo "Running test ${counter} / ${n_test_files}"
  pytest -m "not internal" -vv --cov-append --cov=bionemo -k "not test_model_training and not test_scripts_e2e" ${testfile}
  test_status=$?
  # Exit code 5 means no tests were collected: https://docs.pytest.org/en/stable/reference/exit-codes.html
  test_status=$(($test_status == 5 ? 0 : $test_status))
  # Updating overall status of tests
  status=$(($test_status > $status ? $test_status : $status))
  set +x
  ((counter++))
done

echo "Waiting for the tests to finish..."
wait

echo "Completed"

exit $status
