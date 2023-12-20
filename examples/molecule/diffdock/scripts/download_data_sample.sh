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

REPO_DIR=examples/tests/test_data/molecule/diffdock
# This unset is necessary because this script may be called by other scripts.
unset DATA_PATH
# Function to display help
display_help() {
    echo "Usage: $0 [-data_path <path>] [-pbss <value>] [-help]"
    echo "  -data_path <path>   Specify the data path, \$BIONEMO_HOME/$REPO_DIR by default"
    echo "  -pbss <value>       If set, data will be download from PBSS. If unset, NGC by default."
    echo "  -help               Display this help message"
    exit 1
}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        -pbss)
            PBSS=true
            shift
            ;;
        -help)
            display_help
            ;;
        *)
            echo "Unknown parameter: $1"
            display_help
            ;;
    esac
done

if [ -n "$DATA_PATH" ]; then
    echo "Data path specified: $DATA_PATH"
elif [ -n "$BIONEMO_HOME" ]; then
    echo "Data path defaulting to \$BIONEMO_HOME repo base: $BIONEMO_HOME/$REPO_DIR"
    DATA_PATH=$BIONEMO_HOME/$REPO_DIR
else
    echo "\$BIONEMO_HOME is unset and -data_path was not provided. Exiting."
    exit 1
fi

if [ -n "$PBSS" ]; then
    echo "Downloading from PBSS to $DATA_PATH"
    aws s3 cp s3://bionemo-ci/test-data/diffdock/diffdock_vprocessed_sample $DATA_PATH --endpoint-url https://pbss.s8k.io --recursive
    aws s3 cp s3://bionemo-ci/test-data/diffdock/diffdock_vpreprocessing_test $DATA_PATH --endpoint-url https://pbss.s8k.io --recursive
    # Add actions for pbss
else
    echo "Downloading from NGC to $DATA_PATH"
    ngc registry resource download-version nvidian/clara-lifesciences/diffdock:processed_sample
    tar -xvf diffdock_vprocessed_sample/diffdock_processsed_sample.tar.gz -C $DATA_PATH
    rm -r diffdock_vprocessed_sample/
    ngc registry resource download-version nvidian/clara-lifesciences/diffdock:preprocessing_test
    tar -xvf diffdock_vpreprocessing_test/preprocessing_test.tar.gz -C $DATA_PATH
    rm -r diffdock_vpreprocessing_test/
fi
