#!/bin/bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

source download_models.sh

current_script="$0"

# Check if the script is executable
if [ ! -x "$current_script" ]; then
    # If not, make it executable
    echo "This script is not executable. Making it executable..."
    chmod +x "$current_script"
    echo "Done."
fi

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

    pull               - pull an existing container
    download           - download pre-trained models
    download_test_data - download data necessary for openfold, diffdock tests.
    download_all       - download_test_data and download pretrained models.
    build              - build a container, only recommended if customization is needed
    run                - launch the docker container in non-dev mode. Code is cloned from git and installed.
    push               - push a container to a registry
    dev                - launch a new container in development mode. Local copy of the code is mounted and installed.
    attach             - attach to a running container


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
    LOCAL_REPO_PATH
        Path on workstation or cluster to code, e.g., /home/user/code/bionemo. Inside the bionemo container,
        this is set to /workspace/bionemo by default.
    DOCKER_REPO_PATH
        Set this to change the location of the library in the container, e.g. for development work.
        It is set to /workspace/bionemo by default and a lot of the examples expect this path to be valid.
    WANDB_API_KEY
        Weights and Balances API key to upload runs to WandB. Can also be uploaded afterwards., e.g. Dkjdf...
        This value is optional -- Weights and Biases will log data and not upload if missing.
    JUPYTER_PORT
        Port for launching jupyter lab, e.g. 8888
    REGISTRY
        Container registry URL. e.g., nvcr.io. Only required to push/pull containers.
    REGISTRY_USER
        container registry username. e.g., '$oauthtoken' for registry access. Only required to push/pull containers.
    DEV_CONT_NAME
        Docker name for development container
    NGC_CLI_API_KEY
        NGC API key -- this is required for downloading models and other assets. Run \`ngc config --help\` for details.
    NGC_CLI_ORG
        NGC organization name. Default is nvidian. Run \`ngc config --help\` for details.
    NGC_CLI_TEAM
        NGC team name.  Run \`ngc config --help\` for details.
    NGC_CLI_FORMAT_TYPE
        NGC cli format. Default is ascii.  Run \`ngc config --help\` for details.
    GITLAB_TOKEN
        gitlab access token, used when build container with wheels stored in gitlab registery

EOF
    exit
}

# Defaults for `.env` file
BIONEMO_IMAGE=${BIONEMO_IMAGE:=nvcr.io/nvidian/cvai_bnmo_trng/bionemo:dev}
LOCAL_REPO_PATH=$(pwd)
DOCKER_REPO_PATH=${DOCKER_REPO_PATH:=/workspace/bionemo}
LOCAL_RESULTS_PATH=${LOCAL_RESULTS_PATH:=${LOCAL_REPO_PATH}/results}
DOCKER_RESULTS_PATH=${DOCKER_RESULTS_PATH:=${DOCKER_REPO_PATH}/results}
LOCAL_DATA_PATH=${LOCAL_DATA_PATH:=${LOCAL_REPO_PATH}/data}
DOCKER_DATA_PATH=${DOCKER_DATA_PATH:=${DOCKER_REPO_PATH}/data}
LOCAL_MODELS_PATH=${LOCAL_MODELS_PATH:=${LOCAL_REPO_PATH}/models}
DOCKER_MODELS_PATH=${DOCKER_MODELS_PATH:=${DOCKER_REPO_PATH}/models}
WANDB_API_KEY=${WANDB_API_KEY:=NotSpecified}
JUPYTER_PORT=${JUPYTER_PORT:=8888}
REGISTRY=${REGISTRY:=NotSpecified}
REGISTRY_USER=${REGISTRY_USER:='$oauthtoken'}
DEV_CONT_NAME=${DEV_CONT_NAME:=bionemo}
NGC_CLI_API_KEY=${NGC_CLI_API_KEY:=NotSpecified}
NGC_CLI_ORG=${NGC_CLI_ORG:=nvidian}
NGC_CLI_TEAM=${NGC_CLI_TEAM:=NotSpecified}
NGC_CLI_FORMAT_TYPE=${NGC_CLI_FORMAT_TYPE:=ascii}


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
    echo LOCAL_REPO_PATH=${LOCAL_REPO_PATH} \# This needs to be set to BIONEMO_HOME for local \(non-dockerized\) use >> $LOCAL_ENV
    echo DOCKER_REPO_PATH=${DOCKER_REPO_PATH} \# This is set to BIONEMO_HOME in container >> $LOCAL_ENV
    echo LOCAL_RESULTS_PATH=${LOCAL_RESULTS_PATH} >> $LOCAL_ENV
    echo DOCKER_RESULTS_PATH=${DOCKER_RESULTS_PATH} >> $LOCAL_ENV
    echo LOCAL_DATA_PATH=${LOCAL_DATA_PATH} >> $LOCAL_ENV
    echo DOCKER_DATA_PATH=${DOCKER_DATA_PATH} >> $LOCAL_ENV
    echo LOCAL_MODELS_PATH=${LOCAL_MODELS_PATH} >> $LOCAL_ENV
    echo DOCKER_MODELS_PATH=${DOCKER_MODELS_PATH} >> $LOCAL_ENV
    echo WANDB_API_KEY=${WANDB_API_KEY} >> $LOCAL_ENV
    echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV
    echo REGISTRY=${REGISTRY} >> $LOCAL_ENV
    echo REGISTRY_USER=${REGISTRY_USER} >> $LOCAL_ENV
    echo DEV_CONT_NAME=${DEV_CONT_NAME} >> $LOCAL_ENV
    echo NGC_CLI_API_KEY=${NGC_CLI_API_KEY} >> $LOCAL_ENV
    echo NGC_CLI_ORG=${NGC_CLI_ORG} >> $LOCAL_ENV
    echo NGC_CLI_TEAM=${NGC_CLI_TEAM} >> $LOCAL_ENV
    echo NGC_CLI_FORMAT_TYPE=${NGC_CLI_FORMAT_TYPE} >> $LOCAL_ENV
fi

# Default paths for framework. We switch these depending on whether or not we are inside
# a docker environment. It is assumed that if we are in a docker environment, then it's the
# bionemo image built with `setup/Dockerfile`.


if [ -f /.dockerenv ]; then
    echo "Running inside a Docker container, using DOCKER paths from .env file."
    RESULT_PATH=${DOCKER_RESULTS_PATH}
    DATA_PATH=${DOCKER_DATA_PATH}
    MODEL_PATH=${DOCKER_MODELS_PATH}
    BIONEMO_HOME=${DOCKER_REPO_PATH}
else
    echo "Not running inside a Docker container, using LOCAL paths from .env file."
    RESULT_PATH=${LOCAL_RESULTS_PATH}
    DATA_PATH=${LOCAL_DATA_PATH}
    MODEL_PATH=${LOCAL_MODELS_PATH}
    BIONEMO_HOME=${LOCAL_REPO_PATH}
fi

# Additional variables that will be used in the script when sent in the .env file:
# BASE_IMAGE        Custom Base image for building.
# NEMO_HOME         Path to external copy of NeMo source code, which is mounted at /workspace/nemo. This allows a different version of NeMo to be used with code.
# TOKENIZERS_PATH   Workstation directory to be mounted to /tokenizers inside container

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
    --shm-size=4g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e TMPDIR=/tmp/ \
    -e NUMBA_CACHE_DIR=/tmp/ "


# add current git hash as docker image metadata
if git rev-parse --git-dir > /dev/null 2>&1; then
    BIONEMO_GIT_HASH=$(git rev-parse --short HEAD)
else
    BIONEMO_GIT_HASH="not in git repository"
fi

DOCKER_BUILD_CMD="docker build --network host \
    -t ${BIONEMO_IMAGE} \
    --build-arg GITLAB_TOKEN=${GITLAB_TOKEN} \
    --label com.nvidia.bionemo.git_hash='${BIONEMO_GIT_HASH}' \
    -f setup/Dockerfile"


download() {
    mkdir -p ${MODEL_PATH}
    download_bionemo_models "${@}"
}

download_test_data() {
    echo 'Downloading test data for openfold...'
    source $BIONEMO_HOME/examples/protein/openfold/scripts/download_data_sample.sh
    echo 'Openfold data download complete.'
    echo 'Downloading test data for diffdock...'
    source $BIONEMO_HOME/examples/molecule/diffdock/scripts/download_data_sample.sh
    echo 'Diffdock data download complete.'
    echo 'Unzipping ESM2 test data...'
    unzip $BIONEMO_HOME/examples/tests/test_data/uniref202104_esm2_qc_test200_val200.zip -d $BIONEMO_HOME/examples/tests/test_data/
    echo 'ESM2 test data unzipped.'
}



docker_login() {
    local ngc_api_key_is_set_=$(ngc_api_key_is_set)
    if [ $ngc_api_key_is_set_ == true ]; then
        docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${NGC_CLI_API_KEY}
    else
        echo 'Docker login has been skipped. Container pushing and pulling may fail.'
    fi
}


pull() {
    docker_login
    docker pull ${BIONEMO_IMAGE}
    exit
}


build() {
    local IMG_NAME=($(echo ${BIONEMO_IMAGE} | tr ":" "\n"))
    local PACKAGE=0
    local CLEAN=0

    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--pkg)
                PACKAGE=1
                shift
                ;;
            -c|--clean)
                CLEAN=1
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

    if [ ${CLEAN} -eq 1 ]
    then
        DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --no-cache"
    fi

    if [ ! -z "${BASE_IMAGE}" ];
    then
        DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --build-arg BASE_IMAGE=${BASE_IMAGE}"
    fi

    DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} -t ${IMG_NAME[0]}:latest"

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

    docker_login
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
    mkdir -p ${MODEL_PATH}
    DEV_PYTHONPATH=""

    if [ ! -z "${NEMO_HOME}" ];
    then
        DOCKER_CMD="${DOCKER_CMD} -v ${NEMO_HOME}:/workspace/nemo "
        DOCKER_CMD="${DOCKER_CMD} --env NEMO_HOME=${NEMO_HOME} "
        DEV_PYTHONPATH="${DEV_PYTHONPATH}:/workspace/nemo"
    fi

    # Note: For BIONEMO_HOME, if we are invoking docker, this should always be
    # the docker repo path.
    DOCKER_CMD="${DOCKER_CMD} --env BIONEMO_HOME=$DOCKER_REPO_PATH"
    DOCKER_CMD="${DOCKER_CMD} --env WANDB_API_KEY=$WANDB_API_KEY"
    DOCKER_CMD="${DOCKER_CMD} --env NGC_CLI_API_KEY=$NGC_CLI_API_KEY"
    DOCKER_CMD="${DOCKER_CMD} --env NGC_CLI_ORG=$NGC_CLI_ORG"
    DOCKER_CMD="${DOCKER_CMD} --env NGC_CLI_TEAM=$NGC_CLI_TEAM"
    DOCKER_CMD="${DOCKER_CMD} --env NGC_CLI_FORMAT_TYPE=$NGC_CLI_FORMAT_TYPE"

    # For development work
    echo "Mounting ${LOCAL_REPO_PATH} at ${DOCKER_REPO_PATH} for development"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_REPO_PATH}:${DOCKER_REPO_PATH} -e HOME=${DOCKER_REPO_PATH} -w ${DOCKER_REPO_PATH} "
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_RESULTS_PATH}:${DOCKER_RESULTS_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_DATA_PATH}:${DOCKER_DATA_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v ${LOCAL_MODELS_PATH}:${DOCKER_MODELS_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v /etc/passwd:/etc/passwd:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/group:/etc/group:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/shadow:/etc/shadow:ro "
    DOCKER_CMD="${DOCKER_CMD} -u $(id -u):$(id -g) "

    # For dev mode, mount the local code for development purpose
    # and mount .ssh dir for working with git
    if [[ $1 == "dev" ]]; then
        echo "Mounting ~/.ssh up for development"
        DOCKER_CMD="$DOCKER_CMD -v ${HOME}/.ssh:${HOME}/.ssh:ro"

        echo "Prepending ${DOCKER_REPO_PATH} to PYTHONPATH for development"
        DEV_PYTHONPATH="${DOCKER_REPO_PATH}:${DOCKER_REPO_PATH}/generated:${DEV_PYTHONPATH}"
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
    ${DOCKER_CMD} --rm -it --gpus all -e HOME=${DOCKER_REPO_PATH} -w ${DOCKER_REPO_PATH} --name ${DEV_CONT_NAME} ${BIONEMO_IMAGE} ${CMD}
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
    download)
        shift
        download "$@"
        ;;
    download_all)
        shift
        download "$@"
        download_test_data
        ;;
    protoc | build | run | push | pull | dev | attach | download_test_data)
        $@
        ;;
    *)
        usage
        ;;
esac
