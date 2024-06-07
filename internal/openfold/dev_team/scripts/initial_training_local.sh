#!/bin/bash

# title: initial_training_local.sh
# usage:
#   (1) create and enter an environment, e.g. docker image, with needed libraries, with data mounted
#   (2) set parameters
#   (3)
#
#
#   
# description:
#   Run initial training..
#

# ============================================================
# (0) preamble
# ============================================================
MESSAGE_TEMPLATE='********initial_training_local.sh: %s\n'
DATE_OF_SCRIPT=$(date +'%Y%m%dT%H%M%S')
printf "${MESSAGE_TEMPLATE}" "begin"
printf "${MESSAGE_TEMPLATE}" "DATE_OF_SCRIPT=${DATE_OF_SCRIPT}"
printf "${MESSAGE_TEMPLATE}" "whoami=$(whoami)"
printf "${MESSAGE_TEMPLATE}" "hostname=$(hostname)"

# =============================================================
# (1) set run-specific parameters
# =============================================================
CONFIG_NAME='openfold_initial_training' # name of the yaml config file with parameters

# Input data
DATA_PATH_IN_CONTAINER=${BIONEMO_HOME}/examples/tests/test_data/openfold_data

# Training parameters
NUM_STEPS_IN_ONE_EPOCH=4
VAL_CHECK_INTERVAL=4
MAX_STEPS=6
MAX_EPOCHS=2

# Output directory and logging
PROJECT_NAME="openfold-initial-training"  # project name, will be used for logging
EXP_TAG="${DATE_OF_SCRIPT}"  # any additional experiment info, can be empty
EXP_NAME="openfold_${EXP_TAG}"
CREATE_WANDB_LOGGER=False
EXP_DIR=${BIONEMO_HOME}/result/${PROJECT_NAME}/${EXP_NAME}

# ================================
# (2) Sanity checks and early exit
# ================================
if [[ -z "${AWS_SECRET_ACCESS_KEY}" ]]; then
    printf "${MESSAGE_TEMPLATE}" "AWS_SECRET_ACCESS_KEY is empty, please add to env and rerun"
    exit
fi

# =========================================
# (3) Make result directory and set permissions
# =========================================
mkdir -p ${EXP_DIR}

# =========================
# (4) Create command string
# =========================
read -r -d '' COMMAND <<EOF
echo "---------------" &&
echo "Starting training" &&
export HYDRA_FULL_ERROR=1 &&
export NEMO_TESTING=1 &&
cd \$BIONEMO_HOME &&
if [[ ! -d "${DATA_PATH_IN_CONTAINER}" ]]; then
    export AWS_ENDPOINT_URL=https://pbss.s8k.io
    export AWS_ACCESS_KEY_ID=team-bionemo 
    ./examples/protein/openfold/scripts/download_data_sample.sh -data_path ${BIONEMO_HOME}/examples/tests/test_data/ -pbss
fi &&
if [[ ! -f /usr/local/bin/hhalign ]]; then
    examples/protein/openfold/scripts/install_third_party.sh 
fi &&
python examples/protein/openfold/train.py \\
    --config-name=${CONFIG_NAME} \\
    ++model.data.dataset_path=${DATA_PATH_IN_CONTAINER} \\
    ++model.num_steps_in_one_epoch=${NUM_STEPS_IN_ONE_EPOCH} \\
    ++trainer.num_nodes=1 \\
    ++trainer.devices=1 \\
    ++trainer.val_check_interval=${VAL_CHECK_INTERVAL} \\
    ++trainer.max_steps=${MAX_STEPS} \\
    ++trainer.max_epochs=${MAX_EPOCHS} \\
    ++trainer.accumulate_grad_batches=1 \\
    ++exp_manager.exp_dir=${EXP_DIR} \\
    ++exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \\
    &> train_${DATE_OF_SCRIPT}.log;
    
EOF

# =========================
# Execute command 
# ==========================
eval "${COMMAND}"

# =========================
# Post-ample command 
# ==========================
printf "${MESSAGE_TEMPLATE}" "end with success"

#set +eu
