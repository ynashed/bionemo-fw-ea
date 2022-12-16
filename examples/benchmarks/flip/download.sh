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

# This script is used to run inference on a single GPU using the ESM-1b model.

FLIP_REMOTE_DATA=http://data.bioembeddings.com/public/FLIP/fasta/all_fasta.zip
DATA_PATH=/data
FLIP_DATA_PATH=${DATA_PATH}/flip

wget -c ${FLIP_REMOTE_DATA} -P ${FLIP_DATA_PATH}
unzip ${FLIP_DATA_PATH}/all_fasta.zip -d ${FLIP_DATA_PATH}
rm -R ${FLIP_DATA_PATH}/__MACOSX
mv ${FLIP_DATA_PATH}/all_fasta/* ${FLIP_DATA_PATH}/
rm -R ${FLIP_DATA_PATH}/all_fasta