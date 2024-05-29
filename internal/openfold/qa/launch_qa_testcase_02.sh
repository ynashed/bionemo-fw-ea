#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --partition=interactive
#SBATCH 
#SBATCH --nodes=1
#SBATCH --gpus-per-node 8
#SBATCH --time 01:00:00                 # wall time
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --exclusive
#SBATCH --comment=

#
# title: launch_qa_testcase_02.sh
# description: fine tuning, single node
#
# usage:
#   (1) update the parameter IMAGE_NAME in this file
#   (2) run command
#         sbatch path-to-script/launch_qa_testcase_02.sh
#
# notes:
#   (1) Default SBATCH variable assignments must appear immediately after the /bin/bash statement.
#   (2) The user may need to provide an account name on line 2.
#   (3) We use the interactive queue, for shorter queue times
#
# dependencies
#   (1) dataset at ${INPUT_DIR} is needed
#   (2) AWS and NGC credentials are needed
#
# tests:
#   (a) loads a checkpoint
#   (b) single epoch of training
#   (c) validation step occurs
#   (d) writes checkpoint file
#
# expected results / success criteria:
#   (1) There is a single slurm job
#   (2) Estimated run time: Once the job is started, it should take less than 10 minutes.
#   (3) Users should obtain a checkpoint file, in the directory ${OUTPUT_DIR}/artifacts/checkpoints, called
#       openfold--multisessionstep=6.0--step=6--val_lddt_ca=*.ckpt     
#
# updated / reviewed: 2024-05-22
#

# (0) preamble
MESSAGE_TEMPLATE='********launch_qa_testcase_02.sh: %s\n'
DATETIME_SCRIPT_START=$(date +'%Y%m%dT%H%M%S')

printf "${MESSAGE_TEMPLATE}" "begin at datetime=${DATETIME_SCRIPT_START}"
set -xe

# (1) set some task-specific parameters
IMAGE_NAME=nvcr.io/nvidian/cvai_bnmo_trng/bionemo:pbinder
OUTPUT_DIR=/lustre/fsw/portfolios/convai/users/pbinder/qa/testcase_02_${DATETIME_SCRIPT_START}
INPUT_DIR=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/openfold/openfold_from_tgrzegorzek_20240228


# (2) create output directories
mkdir -p ${OUTPUT_DIR}/logs; mkdir -p ${OUTPUT_DIR}/artifacts
ls ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

# (3) print JOBID.
echo JOBID $SLURM_JOB_ID

# (4) Run the command.
srun --mpi=pmix \
    --export=AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY},AWS_ACCESS_KEY_ID=team-bionemo,NGC_CLI_API_KEY=${NGC_CLI_API_KEY},NGC_API_KEY=${NGC_CLI_API_KEY},NGC_CLI_ORG=nvidian \
    --container-image=${IMAGE_NAME} \
    --output=${OUTPUT_DIR}/slurm-%j.out \
    --error=${OUTPUT_DIR}/error-%j.out \
    --container-mounts=${OUTPUT_DIR}/logs:/result,${OUTPUT_DIR}/artifacts:/result,${INPUT_DIR}:/data \
    bash -c "trap : SIGTERM ; set -x; set -e; echo 'launch_qa_testcase_02.sh - before date' &&
    	 export HYDRA_FULL_ERROR=1 &&
         echo "'date=$(date +'%Y%m%dT%H%M')'" &&
         cd /workspace/bionemo &&
         echo 'launch_qa_testcase_02.sh - before download_artifacts.py' &&
         python download_artifacts.py  --download_dir  /workspace/bionemo --data openfold_training --source pbss --verbose  &&
         python download_artifacts.py --source pbss --download_dir models  --models openfold_initial_training_inhouse &&                                                          
         echo 'launch_qa_testcase_02.sh - after download_artifacts.py' &&                                                                                         
         echo 'launch_qa_testcase_02.sh - before install_third_party.sh' &&                                                                                       
         ./examples/protein/openfold/scripts/install_third_party.sh &&                                                                                                 
         echo 'launch_qa_testcase_02.sh - after install_third_party.sh' &&
         echo 'launch_qa_testcase_02.sh - before train.py' &&
         python examples/protein/openfold/train.py \
        --config-name openfold_finetuning \
        restore_from_path=models/protein/openfold/openfold_initial_training_inhouse_checkpoint.nemo \
        model.data.dataset_path=/data \
        trainer.num_nodes=1 \
        trainer.devices=8 \
        trainer.max_steps=60 \
        trainer.val_check_interval=20 \
        ++model.data.prepare.create_sample=False \
        exp_manager.exp_dir=/result &&
	echo "'date=$(date +'%Y%m%dT%H%M')'" &&
  	echo 'launch_qa_testcase_02.sh - after everything'
	"
set +x
printf "${MESSAGE_TEMPLATE}" "end with success"


