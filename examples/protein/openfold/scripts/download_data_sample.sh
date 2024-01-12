#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

REPO_DIR=examples/tests/test_data/
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
    aws s3 cp s3://bionemo-ci/test-data/openfold/openfold_vprocessed_sample $DATA_PATH --endpoint-url https://pbss.s8k.io --recursive
    # Add actions for pbss
else
    echo "Downloading from NGC to $DATA_PATH"
    ngc registry resource download-version nvidian/cvai_bnmo_trng/openfold:processed_sample
    tar -xvf openfold_vprocessed_sample/openfold_sample_data.tar.gz -C $DATA_PATH
    rm -r openfold_vprocessed_sample/
fi





