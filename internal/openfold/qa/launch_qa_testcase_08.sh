#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --partition=interactive
#SBATCH 
#SBATCH -N 2
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-node 8
#SBATCH --time 01:00:00                 # wall time
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --exclusive
#SBATCH --comment=

#
# title: launch_qa_testcase_08.sh
# usage:
#   (1) export AWS_SECRET_ACCESS_KEY at bash shell
#   (2) update the parameter IMAGE_NAME in this file
#   (3) run command:
#   sbatch path-to-script/launch_qa_testcase_08.sh

# notes:
#   (1) Default SBATCH variable assignments must appear immediately after the /bin/bash statement
#   (2) The user may need to provide an account name on line 2
#
#
# expected behavior / success criteria:
#   (1) There is a multi-node slurm job
#   (2) Users should obtain a checkpoint file, in the directory ${OUTPUT_DIR}/artifacts/checkpoints, called
#
#       openfold--multisessionstep=0--step=50--val_lddt_ca=*-last.ckpt
#
# updated / reviewed: 2024-05-22
#

# (0) preamble
MESSAGE_TEMPLATE='********launch_qa_testcase_08.sh: %s\n'
DATETIME_SCRIPT_START=$(date +'%Y%m%dT%H%M%S')
printf "${MESSAGE_TEMPLATE}" "begin at datetime=${DATETIME_SCRIPT_START}"
set -xe

# (1) set some task-specific parameters
IMAGE_NAME=nvcr.io/nvidian/cvai_bnmo_trng/bionemo:pbinder
INPUT_DIR=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/openfold/openfold_from_tgrzegorzek_20240228
OUTPUT_DIR=/lustre/fsw/portfolios/convai/users/pbinder/qa/testcase_08_${DATETIME_SCRIPT_START}


# (2) create output directories
mkdir -p ${OUTPUT_DIR}/logs; mkdir -p ${OUTPUT_DIR}/artifacts
ls ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

# (3) print JOBID.
echo JOBID $SLURM_JOB_ID

# (4) Run the command.
srun --mpi=pmix \
  --export=AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY},AWS_ACCESS_KEY_ID=team-bionemo \
  --container-image=${IMAGE_NAME} \
  --output=${OUTPUT_DIR}/slurm-%j.out \
  --error=${OUTPUT_DIR}/error-%j.out \
  --container-mounts=${OUTPUT_DIR}/logs:/result,${OUTPUT_DIR}/artifacts:/result,${INPUT_DIR}:/data \
  bash -c "trap : SIGTERM ; set -x; set -e; echo 'launch_qa_testcase_08.sh - before date' &&
  echo "'date=$(date +'%Y%m%dT%H%M')'" &&
  export HYDRA_FULL_ERROR=1 &&
  cd /workspace/bionemo &&
  echo 'launch_qa_testcase_08.sh - before install_third_party.sh' &&  
  ./examples/protein/openfold/scripts/install_third_party.sh &&
  echo 'launch_qa_testcase_08.sh - after install_third_party.sh' &&  
  echo 'launch_qa_testcase_08.sh - before download_artifacts.py' &&  
  python download_artifacts.py --source pbss --download_dir models --models openfold_initial_training_inhouse &&
  echo 'launch_qa_testcase_08.sh - after download_artifacts.py' &&
  echo 'launch_qa_testcase_08.sh - before train.py' &&
  python examples/protein/openfold/train.py \
    --config-name openfold_initial_training \
    ++model.data.dataset_path=/data \
    ++model.data.prepare.create_sample=False \
    ++model.optimisations=[layernorm_inductor,layernorm_triton,mha_triton] \
    ++trainer.num_nodes=2 \
    ++trainer.devices=8 \
    ++trainer.max_steps=50 \
    ++trainer.val_check_interval=200 \
    ++trainer.precision=bf16-mixed \
    ++exp_manager.exp_dir=/result \
    ++exp_manager.create_wandb_logger=False &&
  echo "'date=$(date +'%Y%m%dT%H%M')'" &&
  echo 'launch_qa_testcase_08.sh - after everything'"

set +x
printf "${MESSAGE_TEMPLATE}" "end with success"
