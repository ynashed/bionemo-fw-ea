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

source artifact_paths

ngc_api_key_is_set() {
    if [ ! -z ${NGC_CLI_API_KEY} ] && [ ${NGC_CLI_API_KEY} != 'NotSpecified' ]; then
        echo true
    else
        echo false
    fi
}


function setup_model() {
    local model_source=$1
    local base_directory=$2
    local model_target=${base_directory}/$3

    set -e

    echo "Downloading model ${model_source} to ${model_target}..."

    local tmp_root=`mktemp -d`
    local tmp_download_loc="${tmp_root}/bionemo_downloads"
    rm -rf ${tmp_download_loc}
    mkdir -p ${tmp_download_loc}

    if [[ ${model_source} = http* ]]; then
        echo ${model_source} "is http"
        wget -q --show-progress ${model_source} -O ${tmp_download_loc}/model.zip
        download_path=$(unzip -o ${tmp_download_loc}/model.zip -d ${base_directory} | grep "inflating:")
        download_path=$(echo ${download_path} | cut -d ":" -f 2)
        model_basename=$(basename ${download_path})
        downloaded_model_file="${base_directory}/${model_basename}"
        rm -rf ${tmp_download_loc}/model.zip
    elif $use_s3; then
        echo "Downloading model ${model_source} from s3/swiftstack..."
        aws s3 cp $model_source $MODEL_PATH
        echo "Saved model to $MODEL_PATH"
        model_basename=$(basename ${model_source})
        downloaded_model_file="${base_directory}/${model_basename}"
    else
        echo "Downloading model ${model_source} from NGC..."
        ngc registry model download-version \
            --dest ${tmp_download_loc} \
            "${model_source}"

        download_path=$(ls ${tmp_download_loc}/*/*.nemo)
        model_basename=$(basename ${download_path%.nemo})
        mv ${download_path} ${base_directory}
        downloaded_model_file="${base_directory}/${model_basename}.nemo"
        rm -rf ${tmp_download_loc}/*
    fi

    echo "Linking  ${model_target} to ${downloaded_model_file}..."
    mkdir -p $(dirname ${model_target})
    ln -frs ${downloaded_model_file} ${model_target}

    # This file is created to record the version of model
    mkdir -p ${base_directory}/version
    touch ${base_directory}/version/${model_source//[\/]/_}.version
    set +e
}


download_bionemo_models() {
    use_s3=false
    # Check if the -pbss argument is passed. This is more robust than just checking the first arg.
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            -pbss)
            use_s3=true
            shift
            ;;
            *)
            shift
            ;;
        esac
    done

    if $use_s3; then
        if [[ -z "${AWS_ENDPOINT_URL}" || -z "${AWS_ACCESS_KEY_ID}" || -z "${AWS_SECRET_ACCESS_KEY}" ]]; then
            echo "One or more of the required AWS environment variables (AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) are not set!"
            exit 1
        fi
        setup_model \
            "${MEGAMOLBART_MODEL_PBSS}" \
            "${MODEL_PATH}" \
            "molecule/megamolbart/megamolbart.nemo"
        setup_model \
            "${ESM1NV_MODEL_PBSS}" \
            "${MODEL_PATH}" \
            "protein/esm1nv/esm1nv.nemo"
        setup_model \
            "${PROTT5NV_MODEL_PBSS}" \
            "${MODEL_PATH}" \
            "protein/prott5nv/prott5nv.nemo"
        setup_model \
            "${EQUIDOCK_DIPS_MODEL_PBSS}" \
            "${MODEL_PATH}" \
            "protein/equidock/equidock_dips.nemo"
        setup_model \
            "${EQUIDOCK_DB5_MODEL_PBSS}" \
            "${MODEL_PATH}" \
            "protein/equidock/equidock_db5.nemo"
        setup_model \
            "${ESM2NV_650M_MODEL_PBSS}" \
            "${MODEL_PATH}" \
            "protein/esm2nv/esm2nv_650M_converted.nemo"
        setup_model \
            "${ESM2NV_3B_MODEL_PBSS}" \
            "${MODEL_PATH}" \
            "protein/esm2nv/esm2nv_3B_converted.nemo"
        setup_model \
            "${OPENFOLD_INH_FINETUNED_MODEL_PBSS}" \
            "${MODEL_PATH}" \
            "protein/openfold/openfold.nemo"
    else
        local ngc_api_key_is_set_=$(ngc_api_key_is_set)
        if [ $ngc_api_key_is_set_ != true ]; then
            echo 'The NGC cli key ($NGC_CLI_API_KEY) is not set correctly. Model download may fail.'
        fi

        mkdir -p ${MODEL_PATH}
        setup_model \
            "${MEGAMOLBART_MODEL}" \
            "${MODEL_PATH}" \
            "molecule/megamolbart/megamolbart.nemo"
        setup_model \
            "${ESM1NV_MODEL}" \
            "${MODEL_PATH}" \
            "protein/esm1nv/esm1nv.nemo"
        setup_model \
            "${PROTT5NV_MODEL}" \
            "${MODEL_PATH}" \
            "protein/prott5nv/prott5nv.nemo"
        setup_model \
            "${EQUIDOCK_DIPS_MODEL}" \
            "${MODEL_PATH}" \
            "protein/equidock/equidock_dips.nemo"
        setup_model \
            "${EQUIDOCK_DB5_MODEL}" \
            "${MODEL_PATH}" \
            "protein/equidock/equidock_db5.nemo"
        setup_model \
            "${ESM2NV_650M_MODEL}" \
            "${MODEL_PATH}" \
            "protein/esm2nv/esm2nv_650M_converted.nemo"
        setup_model \
            "${ESM2NV_3B_MODEL}" \
            "${MODEL_PATH}" \
            "protein/esm2nv/esm2nv_3B_converted.nemo"
        setup_model \
            "${OPENFOLD_INH_FINETUNED_MODEL}" \
            "${MODEL_PATH}" \
            "protein/openfold/openfold.nemo"
    fi
}
