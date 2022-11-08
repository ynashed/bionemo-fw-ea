#!/bin/bash

NGC_INSTALL_DIR="$1"
cd ${NGC_INSTALL_DIR}/ngc-cli

ESM1NV_MODEL="t6a4nuz8vrsr/esm1nv:0.1.0"
PROTT5NV_MODEL="t6a4nuz8vrsr/prott5nv:0.1.0"
MEGAMOLBART_MODEL="https://api.ngc.nvidia.com/v2/models/nvidia/clara/megamolbart_0_2/versions/0.2.0/zip"
PROJECT_PATH=$(pwd)


function download_model() {
    local model_source=$1
    local model_target=$2
    set -e
    echo "Downloading model ${model_source} to ${model_target}..."
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
        download_path=$(./ngc registry model download-version --dest /tmp/bionemo_downloads "${model_source}" | grep 'local_path')
        download_path=$(echo ${download_path} | grep -o '"local_path": "[^"]*' | grep -o '[^"]*' | tail -1)
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


mkdir -p ${PROJECT_PATH}/models/molecule/megamolbart
mkdir -p ${PROJECT_PATH}/models/protein/esm1nv
mkdir -p ${PROJECT_PATH}/models/protein/prott5nv

download_model \
    "${MEGAMOLBART_MODEL}" \
    "${PROJECT_PATH}/models/molecule/megamolbart/megamolbart.nemo"
download_model \
    "${ESM1NV_MODEL}" \
    "${PROJECT_PATH}/models/protein/esm1nv/esm1nv.nemo"
download_model \
    "${PROTT5NV_MODEL}" \
    "${PROJECT_PATH}/models/protein/prott5nv/prott5nv.nemo"

cp ${PROJECT_PATH}/models/molecule/megamolbart/megamolbart.nemo /model/molecule/megamolbart/
cp ${PROJECT_PATH}/models/protein/esm1nv/esm1nv.nemo /model/protein/esm1nv/
cp ${PROJECT_PATH}/models/protein/prott5nv/prott5nv.nemo /model/protein/prott5nv/
ls -l /model
