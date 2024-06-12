#!/usr/bin/env bash

# # Name of the Docker image
# IMAGE_NAME=${IMAGE_NAME:=nvcr.io/nvidian/cvai_bnmo_trng/bionemo}
# IMAGE_TAG=bfw

# local created_at="$(date --iso-8601=seconds -u)"


# # Build the Docker image
# echo "Building Docker image..."
# echo GITLAB_TOKEN=${GITLAB_TOKEN} \# This needs to be created via your gitlab account as a personal access token with API access enabled. >> $LOCAL_ENV

# local created_at="$(date --iso-8601=seconds -u)"
# DOCKER_BUILD_CMD="docker build --network host \
#     --no-cache \
#     -t ${IMAGE_NAME}:${IMAGE_TAG} \
#     --secret id=GITLAB_TOKEN,env=GITLAB_TOKEN \
#     --label com.nvidia.bionemo.short_git_sha=${BIONEMO_SHORT_GIT_HASH} \
#     --label com.nvidia.bionemo.git_sha=${BIONEMO_GIT_HASH} \
#     --label com.nvidia.bionemo.created_at=${created_at} \
#     -f bfw/Dockerfile"

# echo $DOCKER_BUILD_CMD
# $DOCKER_BUILD_CMD .

# echo "Docker build completed successfully."

# # Run the Docker container
# docker run --ipc=host --shm-size=512m --gpus all -it $IMAGE_NAME:$IMAGE_TAG /bin/bash


#!/bin/bash
current_script="$0"

# Check if the script is executable
if [ ! -x "$current_script" ]; then
    # If not, make it executable
    echo "This script is not executable. Making it executable..."
    chmod +x "$current_script"
    echo "Done."
fi

#Name of the Docker image
IMAGE_NAME=${IMAGE_NAME:=nvcr.io/nvidian/cvai_bnmo_trng/bionemo}
IMAGE_TAG=$(git rev-parse HEAD)


#Function to build the Docker image
build() {
    version_ge() {
        # Returns 0 (true) if $1 >= $2, 1 (false) otherwise
        [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
    }

    # Check Docker version
    docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    required_docker_version="23.0.1"

    if ! version_ge "$docker_version" "$required_docker_version"; then
        echo "Error: Docker version $required_docker_version or higher is required. Current version: $docker_version"
        exit 1
    fi

    # Check Buildx version
    buildx_version=$(docker buildx version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    required_buildx_version="0.10.2"

    if ! version_ge "$buildx_version" "$required_buildx_version"; then
        echo "Error: Docker Buildx version $required_buildx_version or higher is required. Current version: $buildx_version"
        exit 1
    fi


    local created_at="$(date --iso-8601=seconds -u)"
    echo "Building Docker image..."
    local DOCKER_BUILD_CMD="docker buildx build \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        --cache-to type=inline \
	--cache-from nvcr.io/nvidian/cvai_bnmo_trng/bionemo:bionemo2 \
        -f ./Dockerfile"
    echo $DOCKER_BUILD_CMD
    $DOCKER_BUILD_CMD .
    echo "Docker build completed successfully."
}

#Function to run the Docker container
run() {
    echo "running docker container"
    docker run --ipc=host --net=host --shm-size=512m --gpus all --rm -it --entrypoint /bin/bash ${IMAGE_NAME}:${IMAGE_TAG} 
}

case "$1" in
    build)
        build
        ;;
    run)
        run
        ;;
    *)
        echo "Usage: $0 {build|run}"
        exit 1
        ;;
esac
