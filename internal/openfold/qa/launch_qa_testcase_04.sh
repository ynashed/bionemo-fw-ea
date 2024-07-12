#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --job-name=openfold_qa_testcase_04
#SBATCH --partition=interactive
#SBATCH
#SBATCH --nodes=1
#SBATCH --gpus-per-node 8
#SBATCH --time 01:00:00                 # wall time
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --exclusive
#SBATCH --comment=

#
# title: launch_qa_testcase_04.sh
# description: create data sample, single node
#
# usage:
#   (1) export AWS_SECRET_ACCESS_KEY at bash shell
#   (2) update the parameter IMAGE_NAME in this file
#   (3) run command:
#         sbatch path-to-script/launch_qa_testcase_04.sh
#
# notes:
#   (1) Default SBATCH variable assignments must appear immediately after the /bin/bash statement.
#   (2) The user may need to provide an account name on line 2.
#   (3) We use the interactive queue, for shorter queue times
#
# dependencies
#   (1) dataset at ${INPUT_DIR} is needed
#   (2) AWS credentials are needed.
#
# tests:
#   (a) job has access to openfold dataset
#   (b) train.py can create a sample dataset
#
# expected results / success criteria:
#   (1) Estimated run time:
#       after entering the RUNNING state, the job's runtime should be ~<3min
#   (2) At ${OUTPUT_DIR}/data_out/openfold_qa_testcase_04/ you should find the
#       directories and files:
#
#       - open_protein_set
#           - qa-variant
#               - pdb_alignments
#               - uniclust30_alignments
#               - uniclust30_targets
#        - pdb_mmcif
#           - qa-variant
#             - dicts
#             - chains.csv
#             - obsolete.dat
#
# updated / reviewed: 2024-05-20
#

# (0) preamble
MESSAGE_TEMPLATE='********launch_qa_testcase_04.sh: %s\n'
DATETIME_SCRIPT_START=$(date +'%Y%m%dT%H%M%S')

printf "${MESSAGE_TEMPLATE}" "begin at datetime=${DATETIME_SCRIPT_START}"
set -xe

# (1) set some task-specific parameters
#IMAGE_NAME=nvcr.io/nvidian/cvai_bnmo_trng/bionemo:qa_202405_smoke_20240513T1827
IMAGE_NAME=nvcr.io/nvidian/cvai_bnmo_trng/bionemo:mr--996--c557a19b--2024-05-19-qa
INPUT_DIR=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/openfold/openfold_from_tgrzegorzek_20240228
OUTPUT_DIR=/lustre/fsw/portfolios/healthcareeng/users/broland/qa/qa_202405/testcase_04_${DATETIME_SCRIPT_START}

# (2) create output directories
mkdir -p ${OUTPUT_DIR}/logs; mkdir -p ${OUTPUT_DIR}/artifacts; mkdir -p ${OUTPUT_DIR}/data_out
ls ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

# (3) print JOBID.
echo JOBID $SLURM_JOB_ID

# (4) Run the command.
srun --mpi=pmix \
  --export=AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY},AWS_ACCESS_KEY_ID=team-bionemo \
  --container-image=${IMAGE_NAME} \
  --output=${OUTPUT_DIR}/logs/slurm-%j.out \
  --container-mounts=${OUTPUT_DIR}/logs:/result,${OUTPUT_DIR}/artifacts:/result,${OUTPUT_DIR}/data_out:/data_out,${INPUT_DIR}:/data \
  bash -c "trap : SIGTERM ; set -x; set -e; echo 'launch_qa_testcase_04.sh - before date' &&
  echo "'date=$(date +'%Y%m%dT%H%M')'" &&
  export HYDRA_FULL_ERROR=1 &&
  cd /workspace/bionemo &&
  echo 'launch_qa_testcase_04.sh - before install_third_party.sh' &&
  ./examples/protein/openfold/scripts/install_third_party.sh &&
  echo 'launch_qa_testcase_04.sh - after install_third_party.sh' &&
  echo 'launch_qa_testcase_04.sh - before train.py' &&
  python examples/protein/openfold/train.py \
    ++do_training=false \
    model.data.dataset_path=/data \
    model.data.prepare.create_sample=True \
    model.data.prepare.sample.sample_variant=qa-variant \
    model.data.prepare.sample.output_root_path=/data_out/${SLURM_JOB_NAME} \
    exp_manager.exp_dir=/data_out/${SLURM_JOB_NAME} &&
  echo "'date=$(date +'%Y%m%dT%H%M')'" &&
  echo 'launch_qa_testcase_04.sh - after everything'"

set +x
printf "${MESSAGE_TEMPLATE}" "end with success"
