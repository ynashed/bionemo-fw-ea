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

####
# Example shell script to launch training of 
# secondary structure prediction with 
# ProtT5nv or ESM1nv models on NGC. 
# Encoder can be frozen or unfrozen.
####

# TODO: parameters must be adjusted for customer training
# TODO: this script assumes that model checkpoint is already in the container

LOCAL_ENV=.env

CONTAINER_IMAGE=nvidian/clara-lifesciences/bionemo-dev-camirr:latest 
 
# Model architecture -- should be megamolbart
MODEL=megamolbart

#Experiment description
DWNSTRM_TASK=physchem
EXP_TAG="" # customize EXP_TAG to add additional information

# NGC specific parameters
NGC_ARRAY_SIZE=1  #number of nodes for the job
NGC_GPUS_PER_NODE=2 #number of gpus per node
REPLICAS=1 #equal to the number of nodes
ACE=nv-us-west-2
INSTANCE="dgx1v.32g.2.norm" #"dgx1v.16g.8.norm"
ORG=nvidian
TEAM=clara-lifesciences
LABEL=ml__bionemo
WL_LABEL=wl___other___bionemo
JOB_NAME=ml-model.bionemo-fw-${MODEL}-downstream-${DWNSTRM_TASK}-${EXP_TAG}
WORKSPACE= # Your NGC workspace ID goes here

# Model specific parameters
ACCUMULATE_GRAD_BATCHES=1
ENCODER_FROZEN=True
TENSOR_MODEL_PARALLEL_SIZE=1
RESTORE_FROM_PATH=/model/molecule/${MODEL}/${MODEL}.nemo
MICRO_BATCH_SIZE=32

# Dataset and logging parameters
WANDB_API_KEY= # Your personal WandB API key goes here
DATASET_ID=1607058
EXP_DIR=/workspace/nemo_experiments/${MODEL}/phys_chem_finetune_encoder_frozen-${ENCODER_FROZEN}_tp${TENSOR_MODEL_PARALLEL_SIZE}_grad_acc${ACCUMULATE_GRAD_BATCHES}
WANDB_LOGGER_NAME=${MODEL}_phys_chem_finetune_encoder_frozen-${ENCODER_FROZEN}_tp${TENSOR_MODEL_PARALLEL_SIZE}_grad_acc${ACCUMULATE_GRAD_BATCHES}_${EXP_TAG}
CONFIG_PATH=./conf


# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
fi

read -r -d '' COMMAND <<EOF
 python downstream_physchem.py --config-path=${CONFIG_PATH} \
 --config-name=finetune_config exp_manager.exp_dir=${EXP_DIR} \
 exp_manager.wandb_logger_kwargs.offline=False \
 trainer.devices=${NGC_GPUS_PER_NODE} \
 trainer.num_nodes=${NGC_ARRAY_SIZE} \
 model.micro_batch_size=${MICRO_BATCH_SIZE} \
 model.data.dataset_path=/data/physchem/SAMPL \
 model.data.dataset.train=x000 \
 model.data.dataset.val=x000 \
 model.data.dataset.test=x000 \
 exp_manager.wandb_logger_kwargs.name=${WANDB_LOGGER_NAME} \
 trainer.val_check_interval=8 model.global_batch_size=null \
 model.encoder_frozen=${ENCODER_FROZEN} \
 model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
 restore_from_path=${RESTORE_FROM_PATH} \
 trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}
EOF

BCP_COMMAND="bcprun --debug --nnodes=${NGC_ARRAY_SIZE} --npernode=${NGC_GPUS_PER_NODE} -w /workspace/bionemo/examples/molecule/megamolbart -e WANDB_API_KEY=${WANDB_API_KEY} --cmd '"${COMMAND}"'"

#Add --array-type "PYTORCH" to command below for multinode jobs
echo "ngc batch run --name "${JOB_NAME}" --priority NORMAL --preempt RUNONCE --total-runtime 2h --ace "${ACE}" --instance "${INSTANCE}" --commandline "\"${BCP_COMMAND}"\" --result /result/ngc_log --replicas "${REPLICAS}" --image "${CONTAINER_IMAGE}" --org ${ORG} --team ${TEAM} --workspace ${WORKSPACE}:/result --datasetid ${DATASET_ID}:/data/physchem/SAMPL/ --label ${LABEL} --label ${WL_LABEL}" | bash

