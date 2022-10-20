#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####
# Test for ProtT5nv training.
####

### CONFIG ###
BIONEMO_PATH=${BIONEMO_PATH:=$BIONEMO_HOME} # can also be explicity set, e.g. /opt/nvidia/bionemo or /workspace/bionemo
### END CONFIG ###

CONFIG_FILE=prott5nv_test
DATA_MOUNT=${BIONEMO_PATH}/examples/tests/test_data
RESULTS_MOUNT=/tmp/results

TRAINING_ARGS="model.data.dataset_path=${DATA_MOUNT}"

execute() {
    set -x
    python ${BIONEMO_PATH}/examples/protein/prott5nv/pretrain.py \
        --config-path=${BIONEMO_PATH}/examples/tests \
        --config-name=${CONFIG_FILE} \
        exp_manager.exp_dir=${RESULTS_MOUNT} \
        ${TRAINING_ARGS} \
        ++exp_manager.create_wandb_logger="False" \
        ++exp_manager.wandb_logger_kwargs.offline="True" >& ${RESULTS_MOUNT}/cmdline_prints
    set +x
}


train() {
    DO_TRAINING="True"
    execute
}

rm -rf ${RESULTS_MOUNT}
mkdir -p ${RESULTS_MOUNT}
echo "Setting BIONEMO_PATH to ${BIONEMO_PATH}"
export BIONEMO_PATH=${BIONEMO_PATH}

if [[ ${BIONEMO_PATH} != ${BIONEMO_HOME} ]]; then
    echo "Prepending ${BIONEMO_PATH} to PYTHONPATH for development"
    DEV_PYTHONPATH=${BIONEMO_PATH}
else
    DEV_PYTHONPATH=""
fi
export PYTHONPATH="${DEV_PYTHONPATH}:${BIONEMO_WORKSPACE}:${BIONEMO_WORKSPACE}/generated:$PYTHONPATH"

export HYDRA_FULL_ERROR=1
cd ${BIONEMO_PATH}/examples

if [ $# -eq 0 ]; then
    ARGS=train
    CMD='train'
else
    ARGS=$1
    CMD=$@
fi

case $ARGS in
    train)
        $CMD
        ;;
    *)
        usage
        exit 1
        ;;
esac

RES=$(cat ${RESULTS_MOUNT}/cmdline_prints | grep "Epoch 0:  99%" | sed 's/^.*\ loss/loss/' | sed 's/.$//') 
arrRes=(${RES//,/ })
echo ${arrRes[0]}" "${arrRes[2]}" "${arrRes[3]}" "${arrRes[4]} >& ${RESULTS_MOUNT}/result_log

DIFF=$(diff ${RESULTS_MOUNT}/result_log ${BIONEMO_PATH}/examples/tests/expected_results/prott5nv_log)
DIFF_PASSED=$?
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
if [ $DIFF_PASSED -ne 0 ]
then
    echo -e "${RED}FAIL${NC}"
    echo "$DIFF"
    exit 1
else
    echo -e "${GREEN}PASS${NC}"
fi

