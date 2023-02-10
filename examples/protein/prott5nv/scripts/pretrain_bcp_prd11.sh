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
# Example shell script to launch training on NGC BCP - PRD11
####

NGC_ARRAY_SIZE=4  #number of nodes for the job
NGC_GPUS_PER_NODE=8 #number of gpus per node

read -r -d '' COMMAND <<EOF
export WANDB_API_KEY=592f78db4f2dcefcb23293ecb68d7dd3fee7be2e && cd /workspace/bionemo/examples/protein/prott5nv && python pretrain.py         --config-path=conf         --config-name=pretrain_small     do_training=True          model.data.dataset_path=/workspace/bionemo-fw/ntadimeti/uniref202205_0512_1k ++model.data.dataset.train=x000 ++model.data.dataset.val=x000 ++model.data.dataset.test=x000 exp_manager.exp_dir=/workspace/bionemo-fw/ntadimeti/nemo_experiments/prott5nv/pretrain_small ++exp_manager.wandb_logger_kwargs.offline=False ++trainer.devices=8 ++trainer.num_nodes=4 model.validation.validation_enabled=False model.micro_batch_size=48 ++exp_manager.wandb_logger_kwargs.name=prott5nv_4node_bs48_tp1 ++trainer.max_steps=50 ++trainer.val_check_interval=50 ++model.global_batch_size=null 
EOF

BCP_COMMAND="bcprun --debug --nnodes=${NGC_ARRAY_SIZE} --npernode=${NGC_GPUS_PER_NODE} --cmd '"${COMMAND}"'"


ngc batch run --name "bionemo-fw-prott5nv-pretrain" --priority NORMAL --preempt RUNONCE --total-runtime 2h --ace nv-us-west-2 --instance dgxa100.40g.8.norm --commandline "\"${BCP_COMMAND}"\" --result /results --array-type "PYTORCH" --replicas "4" --image "nvidian/bionemo/bionemo-fw:latest" --org nvidian --team bionemo --datasetid 110553:/data/uniref50 --datasetid 110556:/data/zinc_csv --workspace gyjDJygLRoqhhPsWVgFE0g:/workspace/bionemo-fw:RW --label prott5nv_pretrain --order 50


