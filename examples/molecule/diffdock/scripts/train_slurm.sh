#!/bin/bash
#SBATCH --account=??                  # account (user must belong to account)
#SBATCH --nodes=??                    # number of nodes
#SBATCH --partition=??                # partition (should be compatible with account)
#SBATCH --ntasks-per-node=??          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=??
#SBATCH --time=??                     # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --mem=0                       # all mem avail
#SBATCH --mail-type=FAIL              # only send email on failure
#SBATCH --overcommit
#SBATCH --exclusive                   # exclusive node access

# Below is a sample set of parameters for launching DiffDock model training with BioNeMo on SLURM-based clusters
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/molecule/diffdock/conf/train_score.yaml and 
# examples/molecule/diffdock/conf/train_confidence.yaml

# set your Weights and Biases API KEY
if [[ "${WANDB_API_KEY}" == "" ]]; then
  echo "ERROR: Must set WANDB_API_KEY"
  exit 1
fi

if [[ "${BIONEMO_IMAGE}" == "" ]]; then
  echo "ERROR: Must set a specific BIONEMO_IMAGE. This image name must be complete and the image must reside on NGC."
  exit 1
fi

if [[ "${SLURM_JOB_NUM_NODES}" == "" ]]; then
  echo "ERROR: Must set SLURM_JOB_NUM_NODES"
  exit 1
fi

# Directory with data for model training
if [[ "${DATA_PATH}" == "" ]]; then
  echo "ERROR: Must set DATA_PATH. This is where the data lives in the cluster."
  exit 1
fi

# directory to store logs, checkpoints and results
if [[ "${RESULTS_PATH}" == "" ]]; then
  echo "ERROR: Must set RESULTS_PATH. This is where logs, checkpoints, and results will be stored in the cluster."
  exit 1
fi

set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-"score"} # "score", "confidence", to train score or confidence model
if [[ "${MODEL_TYPE}" != "score" && "${MODEL_TYPE}" != "confidence" ]]; then
  echo "ERROR: Invalid MODEL_TYPE, only 'score' or 'confidence' accepted. Found: ${MODEL_TYPE}"
  exit 1
fi

CONFIG_NAME="train_${MODEL_TYPE}" # name of the yaml config file with parameters 

# Training parameters
# =========================
# micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory.
# Suggested value for small span model for A100 80GB is 32
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-12}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-1} # gradient accumulation
MAX_STEPS=${MAX_STEPS:-1000000} # duration of training as the number of training steps
# =========================

# Logging
# =========================
PROJECT_NAME="diffdock_${MODEL_TYPE}_training"  # project name, will be used for logging
EXP_TAG=${EXP_TAG:-"-large"} # any additional experiment info
EXP_NAME="diffdock_${MODEL_TYPE}_batch${MICRO_BATCH_SIZE}_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes${SLURM_JOB_NUM_NODES}${EXP_TAG}"
CREATE_WANDB_LOGGER=True
WANDB_LOGGER_OFFLINE=False
# =========================

# Note: BIONEMO_HOME is set inside the container to the correct repo path (typically /workspace/bionemo)
BIONEMO_HOME==${BIONEMO_HOME:/workspace/bionemo}

# Mounts
# =========================
DATA_MOUNT="${BIONEMO_HOME}/data" # where data will be mounted in the container
RESULTS_MOUNT="${BIONEMO_HOME}/results" # directory where results folder will be mounted in the container

mkdir -p ${RESULTS_PATH}

MOUNTS="${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
# =========================

# NeMo and BioNeMo code is picked from the container. To use code from a shared
# folder instead, please NEMO_CODE and BIONEMO_CODE in the properties file.
if [ ! -z "${NEMO_CODE}" ];
then
    MOUNTS="${MOUNTS},${NEMO_CODE}:/opt/nvidia/nemo"
fi

if [ ! -z "${BIONEMO_CODE}" ];
then
    MOUNTS="${MOUNTS},${BIONEMO_CODE}:$BIONEMO_HOME"
fi


# Necessary Exports
# =========================
export HYDRA_FULL_ERROR=1
# =========================

set -x
srun \
    --output slurm-%j-%n.out \
    --error error-%j-%n.out \
    --container-image ${BIONEMO_IMAGE} \
    --container-mounts ${MOUNTS} \
    --container-workdir /workspace/bionemo/examples/molecule/diffdock \
    --export WANDB_API_KEY="${WANDB_API_KEY}" \
    bash -c "cd   \$BIONEMO_HOME/examples/molecule/diffdock ; \
    export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
    python \$BIONEMO_HOME/examples/molecule/diffdock/train.py \
    --config-path=\$BIONEMO_HOME/examples/molecule/diffdock/conf \
    --config-name=${CONFIG_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.offline=${WANDB_LOGGER_OFFLINE} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.devices=${SLURM_NTASKS_PER_NODE} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
    model.micro_batch_size=${MICRO_BATCH_SIZE}
set +x

