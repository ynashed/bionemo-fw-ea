#!/bin/bash
# For internal use only. It is employed in BioNeMo CI.
# Attempts to download available checkpoints of all BioNeMo models
# USAGE:
# downloading all models: ./internal/scripts/download_models_unit_tests.sh 5 30
# downloading only particular model: ./internal/scripts/download_models_unit_tests.sh 5 30 megamolbart

source download_models.sh

# number of retries
retries=$1
wait_time_between_attempts=$2
# name of the model to download
model_name=$3

for i in $(seq 1 $retries); do
  download_bionemo_models $model_name
  if [ -z "$model_name" ]  || [ "$model_name" = "megamolbart_retro" ]; then
    setup_model "nvidian/clara-lifesciences/megamolbart_retro:0.1.0" "${MODEL_PATH}" "molecule/megamolbart/megamolbart_retro.nemo"
  fi
  ret_value=$?
  [ $ret_value -eq 0 ] && break
  echo "> failed with $ret_value, waiting to retry..."
  sleep $wait_time_between_attempts

done

exit $ret_value