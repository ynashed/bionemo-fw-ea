#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.
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

# Below is a sample set of parameters for launching ESM-1nv or ProtT5nv finetuning for a secondary structure predition 
# downstream task with BioNeMo on BCP clusters
# Replace all ?? with appropriate values prior to launching a job
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/protein/prott5nv/conf/finetune_config.yaml for ProtT5nv model
# or in examples/protein/esm1nv/conf/finetune_config.yaml for ESM-1nv model

LOCAL_ENV=.env
# container must contain pretrained model checkpoints
CONTAINER_IMAGE=??
 
# Model architecture -- can be prott5nv or esm1nv
PROTEIN_MODEL=esm1nv #prott5nv 

# NGC specific parameters
TIME_LIMIT="2h"
NGC_ARRAY_SIZE=1  #number of nodes for the job
NGC_GPUS_PER_NODE=2 #number of gpus per node
REPLICAS=1 #equal to the number of nodes
ACE=nv-us-west-2
INSTANCE="dgx1v.32g.2.norm"
ORG=nvidian
TEAM=clara-lifesciences
LABEL=ml__bionemo
WL_LABEL=wl___other___bionemo 
JOB_NAME=ml-model.bionemo-fw-${PROTEIN_MODEL}-finetune
WORKSPACE=?? # Your NGC workspace ID goes here

# Model specific parameters
ACCUMULATE_GRAD_BATCHES=1
ENCODER_FROZEN=True
TENSOR_MODEL_PARALLEL_SIZE=1
RESTORE_FROM_PATH=/model/protein/${PROTEIN_MODEL}/${PROTEIN_MODEL}.nemo
MICRO_BATCH_SIZE=64

# Dataset and logging parameters
WANDB_API_KEY=?? # Your personal WandB API key goes here
DATASET_ID=1612245 # ID of the dataset with FLIP data
DATA_MOUNT=/data/FLIP # Where FLIP data is mounted in the container
TASK_NAME=secondary_structure # only secondary structure is supported
EXP_DIR=/workspace/nemo_experiments/${PROTEIN_MODEL}/downstream_sec_str_encoder_frozen-${ENCODER_FROZEN}_tp${TENSOR_MODEL_PARALLEL_SIZE}_grad_acc${ACCUMULATE_GRAD_BATCHES}
WANDB_LOGGER_NAME=${PROTEIN_MODEL}_sec_str_finetune_encoder_frozen-${ENCODER_FROZEN}_tp${TENSOR_MODEL_PARALLEL_SIZE}_grad_acc${ACCUMULATE_GRAD_BATCHES}
WANDB_OFFLINE=False # set to True if uploading results to WandB online is undesired
CONFIG_PATH=../${PROTEIN_MODEL}/conf


# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
fi

read -r -d '' COMMAND <<EOF
 python downstream_sec_str.py --config-path=${CONFIG_PATH} \
 --config-name=finetune_config exp_manager.exp_dir=${EXP_DIR} \
 exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE} \
 trainer.devices=${NGC_GPUS_PER_NODE} \
 trainer.num_nodes=${NGC_ARRAY_SIZE} \
 ++model.dwnstr_task_validation.enabled=False \
 model.micro_batch_size=${MICRO_BATCH_SIZE} \
 model.data.task_name=${TASK_NAME} \
 model.data.dataset_path=${DATA_MOUNT}/${TASK_NAME} \
 exp_manager.wandb_logger_kwargs.name=${WANDB_LOGGER_NAME} \
 trainer.val_check_interval=50 model.global_batch_size=null \
 model.encoder_frozen=${ENCODER_FROZEN} \
 model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
 restore_from_path=${RESTORE_FROM_PATH} \
 trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}
EOF

BCP_COMMAND="bcprun --debug --nnodes=${NGC_ARRAY_SIZE} \
             --npernode=${NGC_GPUS_PER_NODE} -w /workspace/bionemo/examples/protein/downstream \
             -e WANDB_API_KEY=${WANDB_API_KEY} --cmd '"${COMMAND}"'"

#Add --array-type "PYTORCH" to command below for multinode jobs
echo "ngc batch run --name "${JOB_NAME}" --priority NORMAL \
      --preempt RUNONCE --total-runtime ${TIME_LIMIT} --ace "${ACE}" \
      --instance "${INSTANCE}" --commandline "\"${BCP_COMMAND}"\" \
      --result /result/ngc_log --replicas "${REPLICAS}" \
      --image "${CONTAINER_IMAGE}" --org ${ORG} --team ${TEAM} \
      --workspace ${WORKSPACE}:/result --datasetid ${DATASET_ID}:${DATA_MOUNT} \
      --label ${LABEL} --label ${WL_LABEL}" | bash

