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
set -x

# Below is a sample set of parameters for launching DSMBind training with BioNeMo on SLURM-based clusters
# Replace all ?? with appropriate values prior to launching a job
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/molecule/dsmbind/conf/train.yaml.

BIONEMO_IMAGE="??" # BioNeMo container image

CONFIG_NAME='train' # name of the yaml config file with parameters

# Note: BIONEMO_HOME is set inside the container to the correct repo path (typically /workspace/bionemo)
BIONEMO_HOME==${BIONEMO_HOME:/workspace/bionemo}

# Mounts
# =========================
DATA_PATH="??" # Directory with data for model training
DATA_FILE=nv_pdbdock_processed.pkl # folder containing data for model training
DATA_MOUNT="${BIONEMO_HOME}/data" # where data will be mounted in the container
RESULTS_PATH="??" # directory to store logs, checkpoints and results
RESULTS_MOUNT="${BIONEMO_HOME}/results" # directory where results folder will be mounted in the container

mkdir -p ${RESULTS_PATH}

MOUNTS="${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"

# Necessary Exports
# =========================
export HYDRA_FULL_ERROR=1
# =========================

# Command
# =========================
read -r -d '' COMMAND <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& echo "Starting training" \
&& cd \$BIONEMO_HOME \
&& cd examples/molecule/dsmbind \
&& python \$BIONEMO_HOME/examples/molecule/dsmbind/train.py \
    --config-path=\$BIONEMO_HOME/examples/molecule/dsmbind/conf \
    --config-name=${CONFIG_NAME} \
    train.num_gpus=${SLURM_GPUS_PER_NODE} \
    train.ckpt_dir=${RESULTS_MOUNT}
    data.processed_training_data_path=${DATA_MOUNT}/${DATA_FILE} \

EOF
# =========================

srun \
    --output slurm-%j-%n.out \
    --error error-%j-%n.out \
    --container-image ${BIONEMO_IMAGE} \
    --container-mounts ${MOUNTS} \
    --container-workdir /workspace/bionemo/examples/molecule/dsmbind \
    bash -c "${COMMAND}"
set +x