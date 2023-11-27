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
# Example shell script to launch training on NGC BCP
####
LOCAL_ENV=.env
CONTAINER_IMAGE=nvcr.io/nvidian/clara-lifesciences/bionemo_equidock:0.1.0

# Model & Dataset
DATASET_NAME=dips
MODEL_NAME=EquiDock_${DATASET_NAME}


# NGC specific parameters
NGC_ARRAY_SIZE=1 #number of nodes for the job
NGC_GPUS_PER_NODE=4 #number of gpus per node
REPLICAS=${NGC_ARRAY_SIZE} # equal to the number of nodes
ACE=nv-us-west-2
INSTANCE="dgx1v.32g.4.norm" # choose instance based on NGC_ARRAY_SIZE and NGC_GPUS_PER_NODE and available GPUs
ORG=nvidian
TEAM=clara-lifesciences
LABEL=ml__bionemo
JOB_NAME=ml-model.${MODEL_NAME}-train
WORKSPACE= # Your NGC workspace ID goes here


# Model specific parameters
MICRO_BATCH_SIZE=32

# Dataset and logging parameters
WANDB_API_KEY= # Your personal WandB API key goes here
DATASET_ID=1610980 # (processed DATASETID: 1610980, raw DATASETID: 1611195 should be processed using examples/protein/equidock/run_preprocess.py)
LR=0.0005
WANDB_LOGGER_NAME=${MODEL_NAME}_lr_${LR}
EXP_NAME=${MODEL_NAME}_nnodes_${NGC_ARRAY_SIZE}_ndevices_${NGC_GPUS_PER_NODE}_bs_${MICRO_BATCH_SIZE}
EXP_DIR=/workspace/nemo_experiments/
CONFIG_PATH=conf

# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
fi

# Note: BIONEMO_HOME is set inside the container to the correct repo path (typically /workspace/bionemo)
read -r -d '' COMMAND <<EOF
  cd \$BIONEMO_HOME/examples/protein/equidock && \
  python pretrain.py  --config-path=conf \
                   --config-name=pretrain   data.data_name=${DATASET_NAME}    trainer.devices=${NGC_GPUS_PER_NODE}    trainer.num_nodes=${NGC_ARRAY_SIZE}    exp_manager.name=${EXP_NAME}    exp_manager.exp_dir=${EXP_DIR}    exp_manager.wandb_logger_kwargs.offline=False    trainer.devices=${NGC_GPUS_PER_NODE}    trainer.num_nodes=${NGC_ARRAY_SIZE}    model.micro_batch_size=${MICRO_BATCH_SIZE}    model.optim.lr=${LR}
EOF

BCP_COMMAND="bcprun --debug --nnodes=${NGC_ARRAY_SIZE} --npernode=${NGC_GPUS_PER_NODE} -w /workspace/bionemo -e WANDB_API_KEY=${WANDB_API_KEY} --cmd '"${COMMAND}"'"

#Add --array-type "PYTORCH" to command below for multinode jobs
echo "ngc batch run --name "${JOB_NAME}" --priority NORMAL --preempt RUNONCE --total-runtime 2h \
     --ace "${ACE}" --instance "${INSTANCE}" --commandline "\"${BCP_COMMAND}"\" \
     --result /result/ngc_log --replicas "${REPLICAS}" --image "${CONTAINER_IMAGE}" \
     --org ${ORG} --team ${TEAM} --workspace ${WORKSPACE}:/result --datasetid ${DATASET_ID}:/data \
     --label ${LABEL}" | bash
