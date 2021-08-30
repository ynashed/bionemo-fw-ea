#!/bin/bash

set -x

##### Development on a cluster with SLURM / Optional interactive or batch training
### CONFIG ###

HOSTNAME=Selene
SLURM_JOB_NUM_NODES=1 # These are used for interactive jobs
SLURM_TASKS_PER_NODE=2
NTASKS=$((${SLURM_JOB_NUM_NODES}*${SLURM_TASKS_PER_NODE}))

ADDITIONAL_FLAGS=" --time 2:00:00 --partition interactive --account swdl --job-name swdl-clara:mgill_megamolbart "
IS_BATCH=0 # 0 for interactive, 1 for sbatch
IS_DEV=1 # 1 will mount code over that in container, 0 does not

PROJECT=MegaMolBART
MEGAMOLBART_CONFIG_FILE=small_span_aug
DATA_FILES_SELECTED=x_OP_000..001_CL_.csv
CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart_training_nemo:210830"

STORAGE_DIR=${HOME}/fs/megatron # ${HOME}/fs is a link to luster fs mount
WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
DATA_DIR=${STORAGE_DIR}/data/zinc_csv_split
CODE_DIR=${STORAGE_DIR}/code/NeMo
OUTPUT_DIR=${STORAGE_DIR}/nemo

### END CONFIG ###

EXP_NAME=${HOSTNAME}_nodes_${SLURM_JOB_NUM_NODES}_gpus_${SLURM_TASKS_PER_NODE}
RESULTS_DIR=${OUTPUT_DIR}/${PROJECT}/${MEGAMOLBART_CONFIG_FILE}/${EXP_NAME}
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out" # Ignored in interactive mode
ERRFILE="${RESULTS_DIR}/error-%j-%n.out" # Ignored in interactive mode

DATA_MOUNT=/data
CODE_MOUNT=/code
OUTPUT_MOUNT=/result
RESULTS_MOUNT=${OUTPUT_MOUNT}/${PROJECT}/${MEGAMOLBART_CONFIG_FILE}/${EXP_NAME}
WORKDIR=${CODE_MOUNT}

MOUNTS="$OUTPUT_DIR:$OUTPUT_MOUNT,$DATA_DIR:$DATA_MOUNT"
if [ $IS_DEV -eq 1 ]; then
    MOUNTS=$MOUNTS",$CODE_DIR:$CODE_MOUNT"
fi

mkdir -p ${RESULTS_DIR}
GPU_LIMIT="$(($SLURM_TASKS_PER_NODE-1))"
SCRIPT_CUDA_VISIBLE_DEVICES=$(seq --separator=',' 0 $GPU_LIMIT)
SCRIPT_PYTHONPATH=${CODE_MOUNT}':$PYTHONPATH'

if [ -z ${WANDB_API_KEY} ]; then
    WANDB_API_KEY=$(grep password $HOME/.netrc | cut -d' ' -f4)
fi

if [ -z ${WANDB_API_KEY} ]; then 
    WANDB_OFFLINE_MODE="true" # handle api key failures gracefully
else
    WANDB_OFFLINE_MODE="false"
fi

read -r -d '' RUN_COMMAND << EOF
echo '*******STARTING********' \
&& echo '---------------' \
&& cd ${CODE_MOUNT}/examples/chem \
&& echo 'Starting training' \
&& export CUDA_VISIBLE_DEVICES=${SCRIPT_CUDA_VISIBLE_DEVICES} \
&& export PYTHONPATH=${SCRIPT_PYTHONPATH} \
&& export HYDRA_FULL_ERROR=1 \
&& export WANDB_API_KEY=${WANDB_API_KEY} \
&& python megamolbart_pretrain.py \
    --config-path=conf \
    --config-name=megamolbart_pretrain_${MEGAMOLBART_CONFIG_FILE} \
    exp_manager.wandb_logger_kwargs.offline=${WANDB_OFFLINE_MODE} \
    exp_manager.wandb_logger_kwargs.job_type=${EXP_NAME} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.gpus=${SLURM_TASKS_PER_NODE} \
    tokenizer.vocab_path=${CODE_MOUNT}/nemo/collections/chem/vocab/megamolbart_pretrain_vocab.txt \
    model.train_ds.filepath=${DATA_MOUNT}/train/${DATA_FILES_SELECTED} \
    model.train_ds.metadata_path=${DATA_MOUNT}/train/metadata.txt \
    model.validation_ds.filepath=${DATA_MOUNT}/val/${DATA_FILES_SELECTED} \
    model.validation_ds.metadata_path=${DATA_MOUNT}/val/metadata.txt \
    model.train_ds.batch_size=128 \
    model.validation_ds.batch_size=128
EOF

SCRIPT_PATH=${RESULTS_DIR}/job_script.sh
echo "${RUN_COMMAND}" > ${SCRIPT_PATH}
export SCRIPT_MOUNT=${RESULTS_MOUNT}/job_script.sh

if [ ${IS_BATCH} -eq 0 ]; then
    ADDITIONAL_FLAGS=${ADDITIONAL_FLAGS}" --pty --nodes ${SLURM_JOB_NUM_NODES} --ntasks ${NTASKS} --ntasks-per-node ${SLURM_TASKS_PER_NODE} "
    EXEC_COMMAND=" bash"
else
    ADDITIONAL_FLAGS="--output $OUTFILE --error $ERRFILE "
    # EXEC_COMMAND=" bash -c ${RUN_COMMAND}"
    EXEC_COMMAND=" bash ${SCRIPT_MOUNT}"
fi

srun $ADDITIONAL_FLAGS \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
--export PYTHONPATH="${SCRIPT_PYTHONPATH}" \
--export SCRIPT_PATH="${SCRIPT_MOUNT}" \
--export WANDB_API_KEY="${WANDB_API_KEY}" \
${EXEC_COMMAND}

set +x
