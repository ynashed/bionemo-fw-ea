#!/bin/bash
#
# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############################################################################
#
# This is my $LOCAL_ENV file
#
LOCAL_ENV=.env
#
###############################################################################

usage() {
    cat <<EOF

USAGE: launch.sh

launch utility script
----------------------------------------

launch.sh [command]

    valid commands:

    pull      - pull an existing container
    download  - download pre-trained models
    build     - build a container, only recommended if customization is needed
    run       - launch the docker container in non-dev mode. Code is cloned from git and installed.
    push      - push a container to a registry
    dev       - launch a new container in development mode. Local copy of the code is mounted and installed.
    attach    - attach to a running container


Getting Started tl;dr
----------------------------------------

    ./launch.sh pull # pull an existing container
    ./launch.sh dev  # launch the container in interactive mode
For more detailed info on getting started, see README.md


More Information
----------------------------------------

Note: This script looks for a file called $LOCAL_ENV in the
current directory. This file should define the following environment
variables:
    BIONEMO_IMAGE
        Container image for BioNeMo training, prepended with registry. e.g.,
        Note that this is a separate (precursor) container from any service associated containers
    PROJECT_MOUNT
        Set this to change the location of the library in the container, e.g. for development work.
        It is set to /workspace/bionemo by default and a lot of the examples expect this path to be valid.
        Use of /workspace/bionemo is strongly recommended as the alternative for development work.
        Only change this if you know what you're doing.
    PROJECT_PATH
        Path on workstation or cluster to code, e.g., /home/user/code/bionemo
    DATA_PATH
        Path on workstation or cluster to data, e.g., /data
    RESULT_PATH
        Path on workstation or cluster to directory for results, e.g. /home/user/results/nemo_experiments
    WANDB_API_KEY
        Weights and Balances API key to upload runs to WandB. Can also be uploaded afterwards., e.g. Dkjdf...
        This value is optional -- Weights and Biases will log data and not upload if missing.
    JUPYTER_PORT
        Port for launching jupyter lab, e.g. 8888
    REGISTRY
        Container registry URL. e.g., nvcr.io. Only required to push/pull containers.
    REGISTRY_USER
        container registry username. e.g., '$oauthtoken' for registry access. Only required to push/pull containers.
    REGISTRY_ACCESS_TOKEN
        container registry access token. e.g., Ckj53jGK... Only required to push/pull containers.
    GITHUB_BRANCH
        Git branch to use for building a container, default is main

EOF
    exit
}


# Don't change these
BIONEMO_HOME=/opt/nvidia/bionemo # Where BioNeMo is installed in container, set the same as Docker container
BIONEMO_WORKSPACE=/workspace/bionemo # Location of examples / config files and where BioNeMo code can be mounted for development


# Defaults for `.env` file
BIONEMO_IMAGE=${BIONEMO_IMAGE:=nvcr.io/t6a4nuz8vrsr/bionemo:latest}
PROJECT_MOUNT=${PROJECT_MOUNT:=/workspace/bionemo}
PROJECT_PATH=${PROJECT_PATH:=$(pwd)}
DATA_PATH=${DATA_PATH:=/tmp}
RESULT_PATH=${RESULT_PATH:=${HOME}/results/nemo_experiments}
WANDB_API_KEY=${WANDB_API_KEY:=NotSpecified}
JUPYTER_PORT=${JUPYTER_PORT:=8888}
REGISTRY=${REGISTRY:=NotSpecified}
REGISTRY_USER=${REGISTRY_USER:='$oauthtoken'}
REGISTRY_ACCESS_TOKEN=${REGISTRY_ACCESS_TOKEN:=NotSpecified}
GITHUB_BRANCH=${GITHUB_BRANCH:=main}

# Model paths
ESM1NV_MODEL="t6a4nuz8vrsr/esm1nv:0.1.0"
PROTT5NV_MODEL="t6a4nuz8vrsr/prott5nv:0.1.0"
MEGAMOLBART_MODEL="https://api.ngc.nvidia.com/v2/models/nvidia/clara/megamolbart_0_2/versions/0.2.0/zip"

# if $LOCAL_ENV file exists, source it to specify my environment
if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
    write_env=0
else
    echo $LOCAL_ENV does not exist. Writing deafults to $LOCAL_ENV
    write_env=1
fi

# If $LOCAL_ENV was not found, write out a template for user to edit
if [ $write_env -eq 1 ]; then
    echo BIONEMO_IMAGE=${BIONEMO_IMAGE} >> $LOCAL_ENV
    echo PROJECT_MOUNT=${PROJECT_MOUNT} >> $LOCAL_ENV
    echo PROJECT_PATH=${PROJECT_PATH} >> $LOCAL_ENV
    echo DATA_PATH=${DATA_PATH} >> $LOCAL_ENV
    echo RESULT_PATH=${RESULT_PATH} >> $LOCAL_ENV
    echo WANDB_API_KEY=${WANDB_API_KEY} >> $LOCAL_ENV
    echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV
    echo REGISTRY=${REGISTRY} >> $LOCAL_ENV
    echo REGISTRY_USER=${REGISTRY_USER} >> $LOCAL_ENV
    echo REGISTRY_ACCESS_TOKEN=${REGISTRY_ACCESS_TOKEN} >> $LOCAL_ENV
    echo GITHUB_BRANCH=${GITHUB_BRANCH} >> $LOCAL_ENV
fi

DATA_MOUNT_PATH="/data"
RESULT_MOUNT_PATH='/result/nemo_experiments'
DEV_CONT_NAME='bionemo'

# Additional variables when send in .env file, is used in the script:
# BASE_IMAGE        Custom Base image for building.
# NEMO_PATH         Path to NeMo source cdoe.
# CHEM_BENCH_PATH   Path to chembench source code. Used for generating benchmark
#                   data
# MODEL_PATH        Local dir to be mounted to /model

# Compare Docker version to find Nvidia Container Toolkit support.
# Please refer https://github.com/NVIDIA/nvidia-docker
DOCKER_VERSION_WITH_GPU_SUPPORT="19.03.0"
if [ -x "$(command -v docker)" ]; then
    DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2'})
fi

PARAM_RUNTIME="--runtime=nvidia"
if [ "$DOCKER_VERSION_WITH_GPU_SUPPORT" == "$(echo -e "$DOCKER_VERSION\n$DOCKER_VERSION_WITH_GPU_SUPPORT" | sort -V | head -1)" ];
then
    PARAM_RUNTIME="--gpus all"
fi

DOCKER_CMD="docker run \
    --network host \
    ${PARAM_RUNTIME} \
    -p ${JUPYTER_PORT}:8888 \
    -v ${DATA_PATH}:${DATA_MOUNT_PATH} \
    -v ${RESULT_PATH}:${RESULT_MOUNT_PATH} \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e TMPDIR=/tmp/ \
    -e NUMBA_CACHE_DIR=/tmp/ "




DOCKER_BUILD_CMD="docker build --network host --ssh default \
    -t ${BIONEMO_IMAGE} \
    --build-arg GITHUB_BRANCH=${GITHUB_BRANCH} \
    --no-cache \
    -f setup/Dockerfile"


function download_model() {
    local model_source=$1
    local model_target=$2
    set -e
    echo "Downloading model ${model_source} to ${model_target}/${model_filename}..."
    local TMP_DOWNLOAD_LOC="/tmp/bionemo_downloads"
    rm -rf ${TMP_DOWNLOAD_LOC}
    mkdir -p ${TMP_DOWNLOAD_LOC}

    if [[ ${model_source} = http* ]]; then
        wget -q --show-progress ${model_source} -O ${TMP_DOWNLOAD_LOC}/model.zip
        download_path=$(unzip -o ${TMP_DOWNLOAD_LOC}/model.zip -d ${PROJECT_PATH}/models | grep "inflating:")
        download_path=$(echo ${download_path} | cut -d ":" -f 2)
        model_basename=$(basename ${download_path})
        downloaded_model_file="${PROJECT_PATH}/models/${model_basename}"
    else
        download_path=$(ngc registry model download-version \
            --dest /tmp/bionemo_downloads \
            "${model_source}" | grep 'Downloaded local path:')

        download_path=$(echo ${download_path} | cut -d ":" -f 2)
        model_basename=$(basename ${download_path} | cut -d "_" -f 1)
        cp ${download_path}/${model_basename}".nemo" ${PROJECT_PATH}/models
        downloaded_model_file="${PROJECT_PATH}/models/${model_basename}.nemo"
    fi
    echo "Linking ${downloaded_model_file} to ${model_target}..."
    cp ${downloaded_model_file} ${model_target}
    # This file is created to record the version of model
    touch ${PROJECT_PATH}/models/${model_source//[\/]/_}.version
    set +e
}


function download() {
    download_model \
        "${MEGAMOLBART_MODEL}" \
        "${PROJECT_PATH}/models/molecule/megamolbart/megamolbart.nemo"
    download_model \
        "${ESM1NV_MODEL}" \
        "${PROJECT_PATH}/models/protein/esm1nv/esm1nv.nemo"
    download_model \
        "${PROTT5NV_MODEL}" \
        "${PROJECT_PATH}/models/protein/prott5nv/prott5nv.nemo"
}


pull() {
    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker pull ${BIONEMO_IMAGE}
    exit
}


build() {
    local IMG_NAME=($(echo ${BIONEMO_IMAGE} | tr ":" "\n"))
    local PACKAGE=0

    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--pkg)
                PACKAGE=1
                shift
                ;;
            -b|--base-image)
                BASE_IMAGE=$2
                shift
                shift
                ;;
            *)
                echo "Unknown option $1. Please --version to specify a version."
                exit 1
                ;;
        esac
    done

    if [ ${PACKAGE} -eq 1 ]
    then
        set -e
        download
        set +e
    fi

    if [ ! -z "${BASE_IMAGE}" ];
    then
        DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --build-arg BASE_IMAGE=${BASE_IMAGE}"
    fi

    DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} -t ${IMG_NAME[0]}:latest"

    # Set up SSH agent for cloning private repo in docker build
    eval "$(ssh-agent -s)"
    find ~/.ssh/ -type f -exec grep -l "PRIVATE" {} \; | xargs ssh-add &> /dev/null

    echo "Building BioNeMo training container..."
    set -x
    DOCKER_BUILDKIT=1 ${DOCKER_BUILD_CMD} .
    set +x
    exit
}


push() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift
                shift
                ;;
            -a|--additional_copies)
                ADDITIONAL_IMAGES="$2"
                shift
                shift
                ;;
            *)
                echo "Unknown option $1. Please --version to specify a version."
                exit 1
                ;;
        esac
    done

    local IMG_NAME=($(echo ${BIONEMO_IMAGE} | tr ":" "\n"))

    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker push ${IMG_NAME[0]}:latest
    docker push ${BIONEMO_IMAGE}

    if [ ! -z "${VERSION}" ];
    then
        docker tag ${BIONEMO_IMAGE} ${IMG_NAME[0]}:${VERSION}
        docker push ${IMG_NAME[0]}:${VERSION}
    fi

    if [ ! -z "${ADDITIONAL_IMAGES}" ];
    then
        IFS=',' read -ra IMAGES <<< ${ADDITIONAL_IMAGES}
        for IMAGE in "${IMAGES[@]}"; do
            docker tag ${BIONEMO_IMAGE} ${IMAGE}
            docker push ${IMAGE}
        done
    fi

    exit
}


setup() {
    mkdir -p ${DATA_PATH}
    mkdir -p ${RESULT_PATH}

    DEV_PYTHONPATH=""

    if [ ! -z "${NEMO_PATH}" ];
    then
        DOCKER_CMD="${DOCKER_CMD} -v ${NEMO_PATH}:/workspace/nemo "
        DEV_PYTHONPATH="${DEV_PYTHONPATH}:/workspace/nemo"
    fi

    if [ ! -z "${CHEM_BENCH_PATH}" ];
    then
        DOCKER_CMD="${DOCKER_CMD} -v ${CHEM_BENCH_PATH}:/workspace/chembench "
        DEV_PYTHONPATH="${DEV_PYTHONPATH}:/workspace/chembench"
    fi

    DOCKER_CMD="${DOCKER_CMD} --env WANDB_API_KEY=$WANDB_API_KEY"
    
    # For development work
    echo "Mounting ${PROJECT_PATH} at ${PROJECT_MOUNT} for development"
    DOCKER_CMD="${DOCKER_CMD} -v ${PROJECT_PATH}:${PROJECT_MOUNT} -e HOME=${PROJECT_MOUNT} -w ${PROJECT_MOUNT} "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/passwd:/etc/passwd:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/group:/etc/group:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/shadow:/etc/shadow:ro "
    DOCKER_CMD="${DOCKER_CMD} -u $(id -u):$(id -g) "
    # For dev use the models in ./models dir
    DOCKER_CMD="${DOCKER_CMD} -v ${PROJECT_PATH}/models:/model"

    # For dev mode, mount the local code for development purpose
    if [[ $1 == "dev" ]]; then
        echo "Prepending ${PROJECT_MOUNT} to PYTHONPATH for development"
        DEV_PYTHONPATH="${PROJECT_MOUNT}:${PROJECT_MOUNT}/generated:${DEV_PYTHONPATH}"  
        DOCKER_CMD="${DOCKER_CMD} --env PYTHONPATH=${DEV_PYTHONPATH}"
    fi
}


dev() {
    CMD='bash'
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--additional-args)
                DOCKER_CMD="${DOCKER_CMD} $2"
                shift
                shift
                ;;
            -t|--tmp)
                DEV_CONT_NAME="${DEV_CONT_NAME}_$2"
                shift
                shift
                ;;
            -d|--demon)
                DOCKER_CMD="${DOCKER_CMD} -d"
                shift
                ;;
            -c|--cmd)
                shift
                CMD="$@"
                break
                ;;
            *)
                echo "Unknown option '$1'.
Available options are -a(--additional-args), -i(--image), -d(--demon) and -c(--cmd)"
                exit 1
                ;;
        esac
    done


    setup "dev"
    set -x
    ${DOCKER_CMD} --rm -it --name ${DEV_CONT_NAME} ${BIONEMO_IMAGE} ${CMD}
    set +x
    exit
}

run() {
    CMD='bash'
    while [[ $# -gt 0 ]]; do
        case $1 in
	    -c|--cmd)
	        shift
		CMD="$@"
		break
		;;
	    *)
	        echo "Unknown option '$1'. Only available option is -c(--cmd)"
		exit 1
		;;
	esac
    done

    set -x
    ${DOCKER_CMD} --rm -it --gpus all -e HOME=${BIONEMO_WORKSPACE} -w ${BIONEMO_WORKSPACE} --name ${DEV_CONT_NAME} ${BIONEMO_IMAGE} ${CMD}
    set +x
    exit
}

attach() {
    set -x
    DOCKER_CMD="docker exec"
    CONTAINER_ID=$(docker ps | grep ${DEV_CONT_NAME} | cut -d' ' -f1)
    ${DOCKER_CMD} -it ${CONTAINER_ID} /bin/bash
    exit
}


protoc() {
    # Generate python stubs for protobuf and grpc services.
    local GEN_DIR="./generated/"
    python3 -m grpc_tools.protoc \
        -I./proto/ \
        --python_out=./generated/ \
        --grpc_python_out=${GEN_DIR} \
        --experimental_allow_proto3_optional \
        ./proto/*.proto
    echo "Generated code is at ${GEN_DIR}."
}


case $1 in
    protoc | download | build | run | push | pull | dev | attach)
        $@
        ;;
    *)
        usage
        ;;
esac
