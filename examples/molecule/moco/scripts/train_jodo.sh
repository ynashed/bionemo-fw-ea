#!/bin/bash
#SBATCH --account=healthcareeng_bionemo                  # account (user must belong to account)
#SBATCH --nodes=1                    # number of nodes
#SBATCH --partition=batch_block1               # partition (should be compatible with account)
#SBATCH --ntasks-per-node=4          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00                     # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --mem=0                       # all mem avail
#SBATCH --mail-type=FAIL              # only send email on failure
#SBATCH --overcommit
#SBATCH --exclusive                   # exclusive node access

# Below is a sample set of parameters for launching MoCo model training with BioNeMo on SLURM-based clusters
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/molecule/moco/conf/train.yaml

BIONEMO_IMAGE="nvcr.io/nvidian/cvai_bnmo_trng/bionemo:moco" # BioNeMo container image
WANDB_API_KEY=1f8cf899b892656eec7d02bc90efd1f97a5bb71c # Add your WANDB API KEY

CONFIG_NAME="train_jodo" 


# Logging
# =========================
EXP_GROUP="filipp-dev"
EXP_NAME="jodo_slurm"
WANDB_ENTITY="clara-discovery"
WANDB_GROUP=${EXP_GROUP}
WANDB_PROJECT="MoCo"
WANDB_MODE="online"
# =========================

# Mounts
# =========================
DATA_MOUNT=/workspace/bionemo/data # where data will be mounted in the container
DATA_PATH="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/fnikitin/data/" # Directory with data for model training and downstream task validation
DATASET=pyg_geom_drug # The processed dataset folder
RESULTS_MOUNT=/workspace/bionemo/results # directory where results folder will be mounted in the container
RESULTS_PATH="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/fnikitin/results/moco/${EXP_GROUP}" # directory to store logs, checkpoints and results
mkdir -p ${RESULTS_PATH}/${EXP_NAME}
CODE_MOUNT=/workspace/bionemo # directory where code folder will be mounted in the container
CODE_PATH="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/fnikitin/code/bionemo" #leave empty if using container bionemo else use path to custom version
if [ -z "$CODE_PATH" ]; then
  MOUNTS="${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
else
  MOUNTS="${CODE_PATH}:${CODE_MOUNT},${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
fi
RESUME_PATH=${RESULTS_PATH}/${EXP_NAME}/checkpoints/last.ckpt
if [ -f "$RESUME_PATH" ]; then
    echo "Resuming from: $RESUME_PATH"
    RESUME_PATH=${RESULTS_MOUNT}/${EXP_NAME}/checkpoints/
else
    RESUME_PATH=null
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
    outdir=${RESULTS_MOUNT} \
    data.dataset_root=${DATA_MOUNT}/${DATASET} \
    run_name=${EXP_NAME} \
    resume=${RESUME_PATH} \
    train.gpus=4"
set +x
