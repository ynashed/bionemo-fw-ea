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
# title: launch_qa_testcase_05.sh
# description: inference using sequences only, single node
#
# usage:
#   (1) update the parameter IMAGE_NAME in this file
#   (2) export AWS_SECRET_ACCESS_KEY and NGC_CLI_API_KEY at bash shell
#   (3) run command

#         sbatch path-to-script/launch_qa_testcase_05.sh
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
#   (a) Load model from checkpoint
#   (b) Run inference on sequences
#   (c) Inference results are outputted
#   
# expected results / success criteria:
#   (1) There is a single slurm job
#   (2) Estimated run time: Once the slurm job has started, the run will take less than 10 minutes
#   (3) Users should obtain 10 files, in the directory ${OUTPUT_DIR}/artifacts/inference-cli, called
#         one.pdb, two.pdb, … ten.pdb, each around 12kb
#       
#
# updated / reviewed: 2024-05-22
#

# (0) preamble
MESSAGE_TEMPLATE='********launch_qa_testcase_05.sh: %s\n'
DATETIME_SCRIPT_START=$(date +'%Y%m%dT%H%M%S')

printf "${MESSAGE_TEMPLATE}" "begin at datetime=${DATETIME_SCRIPT_START}"
set -xe

# (1) set some task-specific parameters
IMAGE_NAME=nvcr.io/nvidian/cvai_bnmo_trng/bionemo:pbinder
OUTPUT_DIR=/lustre/fsw/portfolios/convai/users/pbinder/qa/testcase_05_${DATETIME_SCRIPT_START}
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
    bash -c "trap : SIGTERM ; set -x; set -e; echo 'launch_qa_testcase_05.sh - before date' &&
    	 export HYDRA_FULL_ERROR=1 &&
         echo "'date=$(date +'%Y%m%dT%H%M')'" &&
         cd /workspace/bionemo &&
         echo 'launch_qa_testcase_05.sh - before download_artifacts.py' &&
         python download_artifacts.py --source pbss --download_dir models --models openfold_finetuning_inhouse &&
         echo 'launch_qa_testcase_05.sh - before download_artifacts.py' &&
	 python examples/protein/openfold/infer.py \
	 sequences=\"[HHAGLHHLLCHACLGLHLMG, ALKKMAKHMCKKAGGLCMHL, AHLGMLHMMLHMCGAHCGLL, HAHMMLCHHGLAAHLLKLHA, MMGGLKMLKCCHHHHHKCKC, CCLHMKMCHHAHCGKCGHLL, KHLGLCCGMKAALKKAKHGA, MHKKAHGALMGLGGAGKMML, CCCHCMLMMAMKGKGCGMAA, LAKKALCMCHAKACLLKAMH]\" \
        seq_names=\"[one, two, three, four, five, six, seven, eight, nine, ten]\" \
        restore_from_path=models/protein/openfold/openfold_finetuning_inhouse_checkpoint.nemo \
	results_path=/result/inference-cli \
        model.data.dataset_path=/data \
	++model.data.msa_a3m_filepaths=null \
        trainer.num_nodes=1 \
        trainer.devices=1 \
	++exp_manager.exp_dir=/result			

	echo "'date=$(date +'%Y%m%dT%H%M')'" &&
  	echo 'launch_qa_testcase_05.sh - after everything'
	"
set +x
printf "${MESSAGE_TEMPLATE}" "end with success"


