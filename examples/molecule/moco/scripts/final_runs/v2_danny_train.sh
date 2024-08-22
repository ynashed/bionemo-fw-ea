#!/bin/bash
#SBATCH --account=healthcareeng_bionemo                  # account (user must belong to account)
#SBATCH --nodes=1                    # number of nodes
#SBATCH --partition=grizzly,polar2,polar3,polar4               # partition (should be compatible with account)
#SBATCH --ntasks-per-node=8          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00                     # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --mem=0                       # all mem avail
#SBATCH --mail-type=FAIL              # only send email on failure
#SBATCH --overcommit
#SBATCH --exclusive                   # exclusive node access

# Below is a sample set of parameters for launching MoCo model training with BioNeMo on SLURM-based clusters
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/molecule/moco/conf/train.yaml

BIONEMO_IMAGE="nvcr.io/nvidian/cvai_bnmo_trng/bionemo:moco" # BioNeMo container image
WANDB_API_KEY=40eca22e1f13673f579d3b59efb30d6425fae9e5 # Add your WANDB API KEY

CONFIG_NAME="train_v2_slurm_base" 


# Logging
# =========================
EXP_GROUP="danny-ord-testing"
EXP_NAME="train_v2_slurm_base"
WANDB_ENTITY="clara-discovery"
WANDB_GROUP=${EXP_GROUP}
WANDB_PROJECT="MoCo"
WANDB_MODE="online"
# =========================
USERNAME="dreidenbach"

# Mounts
# =========================
DATA_MOUNT=/data # where data will be mounted in the container
DATA_PATH="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/dreidenbach/data" # Directory with data for model training and downstream task validation
DATASET=pyg_geom_drug # The processed dataset folder
RESULTS_MOUNT=/results # directory where results folder will be mounted in the container
RESULTS_PATH="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/${USERNAME}/results/moco" # directory to store logs, checkpoints and results
mkdir -p ${RESULTS_PATH}/${EXP_NAME}
CODE_MOUNT=/workspace/bionemo # directory where code folder will be mounted in the container
# CODE_PATH="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/${USERNAME}/code/bionemo" #leave empty if using container bionemo else use path to custom version
CODE_PATH="/lustre/fsw/portfolios/healthcareeng/users/${USERNAME}/code/bionemo"
if [ -z "$CODE_PATH" ]; then
  MOUNTS="${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
else
  MOUNTS="${CODE_PATH}:${CODE_MOUNT},${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
fi
RESUME_PATH=${RESULTS_PATH}/${EXP_NAME}/checkpoints/last.ckpt
if [ -f "$RESUME_PATH" ]; then
    echo "Resuming from: $RESUME_PATH"
    RESUME_PATH=${RESULTS_MOUNT}/${EXP_NAME}/checkpoints/last.ckpt
    # Find the largest ema_parameters file
    EMA_PATH=$(find ${RESULTS_PATH}/${EXP_NAME}/checkpoints/ -name 'ema_parameters_epoch_*.pt' | sort -V | tail -n 1)
    EMA_PATH=$(basename "$EMA_PATH")
    EMA_PATH=${RESULTS_MOUNT}/${EXP_NAME}/checkpoints/${EMA_PATH}
else
    RESUME_PATH=null
    EMA_PATH=null
fi
echo "Resuming in container from: $RESUME_PATH and $EMA_PATH"



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
    --container-workdir /workspace/bionemo/examples/molecule/moco \
    --export WANDB_API_KEY="${WANDB_API_KEY}" \
    bash -c "cd   \$BIONEMO_HOME/examples/molecule/moco ; \
    python \$BIONEMO_HOME/examples/molecule/moco/train.py \
    --config-path=\$BIONEMO_HOME/examples/molecule/moco/conf \
    --config-name=${CONFIG_NAME} \
    wandb_params.mode=${WANDB_MODE} \
    wandb_params.entity=${WANDB_ENTITY} \
    wandb_params.group=${WANDB_GROUP} \
    wandb_params.project=${WANDB_PROJECT} \
    data.dataset_root=${DATA_MOUNT}/${DATASET} \
    run_name=${EXP_NAME} \
    resume=${RESUME_PATH} \
    ema_resume=${EMA_PATH} \
    outdir=${RESULTS_MOUNT} \
    train.gpus=8"
set +x
