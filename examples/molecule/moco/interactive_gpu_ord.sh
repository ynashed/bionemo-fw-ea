#!/bin/bash
set -x
SBATCH_ACCOUNT=healthcareeng_bionemo  #convai_bionemo_training #healthcareeng #convai_bionemo_training  # account (user must belong to account)
SBATCH_PARTITION=interactive_singlenode #interactive  # partition (should be compatible with account)
SBATCH_GPUS_PER_NODE=1
SBATCH_TASKS_PER_NODE=1
SBATCH_TIME=04:00:00             # wall time  (8 for batch, backfill, 2 for batch_short)


# Container
# =========================
WANDB_API_KEY=40eca22e1f13673f579d3b59efb30d6425fae9e5 # Add your WANDB API KEY
# DOCKER_IMAGE="nvcr.io/nvidian/cvai_bnmo_trng/proteinfoundation:latest"
# DOCKER_IMAGE="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/dreidenbach/data/protein-foundation-models/containers/proteinfoundation.sqsh"
DOCKER_IMAGE="nvcr.io/nvidian/cvai_bnmo_trng/bionemo:moco"
# Logging
# =========================
EXP_NAME="evaluate_moco"
# =========================

# Mounts
# =========================
USERNAME="dreidenbach"
DATA_MOUNT=/data # where data will be mounted in the container
DATA_PATH="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/${USERNAME}/data" #protein-foundation-models" # The processed dataset folder
CODE_DIR="code/bionemo" ##protein-foundation-models
CODE_MOUNT=/workspace/bionemo # #protein-foundation-models directory where code folder will be mounted in the container
CODE_PATH="/lustre/fsw/portfolios/healthcareeng/users/${USERNAME}/${CODE_DIR}"
RESULTS_MOUNT="/results"
RESULTS_PATH="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/dreidenbach/results/moco"
if [ -z "$CODE_PATH" ]; then
  MOUNTS="${DATA_PATH}:${DATA_MOUNT}"
else
  MOUNTS="${CODE_PATH}:${CODE_MOUNT},${DATA_PATH}:${DATA_MOUNT},${RESULTS_PATH}:${RESULTS_PATH}"
fi
# ========================

srun  \
    --partition=${SBATCH_PARTITION} \
    -A ${SBATCH_ACCOUNT} \
    -t ${SBATCH_TIME} \
    --nodes=1 \
    --gpus-per-node=${SBATCH_GPUS_PER_NODE} \
    --tasks-per-node=${SBATCH_TASKS_PER_NODE} \
    --container-image ${DOCKER_IMAGE} \
    --container-mounts ${MOUNTS} \
    --export WANDB_API_KEY="${WANDB_API_KEY}" \
    --export DATA_PATH="${DATA_MOUNT}" \
    --job-name ${EXP_NAME} \
    --pty \
    /bin/bash
 set +x
