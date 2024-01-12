#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

####
# Example shell script to run ESM1nv training.
####

### CONFIG ###
CONFIG_FILE=pretrain_small
PROJECT=ESM1nv
DATA_MOUNT=/data/uniref2022_05
BIONEMO_HOME=/workspace/bionemo # /workspace/bionemo if library mounted or $BIONEMO_HOME
OUTPUT_MOUNT=/result
WANDB_OFFLINE=True # set to False to upload to WandB while training, otherwise True
RESULTS_MOUNT=${OUTPUT_MOUNT}/nemo_experiments/${PROJECT}/${CONFIG_FILE}
DATA_FILES_SELECTED=x_OP_000..049_CL_  # x000 for a single file for x_OP_000..049_CL_ for a range
### END CONFIG ###

# Don't change these
BIONEMO_HOME=/workspace/bionemo # Location of examples / config files and where BioNeMo code can be mounted for development
RUN_SCRIPT="pretrain.py"
RUN_SCRIPT_DIRECTORY="${BIONEMO_HOME}/examples/protein/esm1nv"
usage() {
cat <<EOF
USAGE: pretrain_quick.sh
ESM1nv pretrain script
----------------------------------------
pretrain_quick.sh [command]
    valid commands:
        preprocess
        train

    default command:
        train

    options:
        -f|--data-files
            List of data files to use without csv file extension
            e.g. x_OP_000..001_CL_ for x000.csv and x001.csv, or can be single file
        -c|--config
            Name of YAML configuration file without file extension
            e.g. esm1nv_config
        -a|--args
            Additional training arguments to be added, repeat flag for additional arguments
            e.g. --args "++trainer.devices=2" --args "++model.tensor_model_parallel_size=2"


EOF
}


execute() {
    TRAINING_ARGS="${TRAINING_ARGS} model.data.dataset_path=${DATA_MOUNT}" # Works even if $TRAINING_ARGS is empty
    TRAINING_ARGS="${TRAINING_ARGS} ++model.data.dataset.train=${DATA_FILES_SELECTED} ++model.data.dataset.val=${DATA_FILES_SELECTED} ++model.data.dataset.test=${DATA_FILES_SELECTED}"
    TRAINING_ARGS="${TRAINING_ARGS} exp_manager.exp_dir=${RESULTS_MOUNT}"
    TRAINING_ARGS="${TRAINING_ARGS} ++exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE}"

    set -x
    python ${RUN_SCRIPT} \
        --config-path=conf \
        --config-name=${CONFIG_FILE} \
	    do_training=${DO_TRAINING} \
        ${TRAINING_ARGS}
    set +x
}


preprocess() {
    DO_TRAINING="False"
    execute
}


train() {
    DO_TRAINING="True"
    execute
}


parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--data-files)
                DATA_FILES_SELECTED=$2
                shift
                shift
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift
                shift
                ;;
            -a|--args)
                TRAINING_ARGS="${TRAINING_ARGS} $2"
                shift
                shift
                ;;
            *)
                usage
                exit 1
                ;;
        esac
    done
}


mkdir -p ${RESULTS_MOUNT}

export HYDRA_FULL_ERROR=1
cd ${RUN_SCRIPT_DIRECTORY}

if [ $# -eq 0 ]; then
    ARGS=train
    parse_args "$@"
else
    ARGS=$1
    shift
    parse_args "$@"
fi

case $ARGS in
    preprocess)
        preprocess
        ;;
    train)
        train
        ;;
    *)
        usage
        exit 1
        ;;
esac
