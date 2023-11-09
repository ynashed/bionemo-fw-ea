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

#!/bin/bash

if [ -n "${PROJECT_MOUNT}" ]
then 
  DATA_PATH=$PROJECT_MOUNT/examples/tests/test_data/molecule/diffdock
else
  DATA_PATH=/workspace/bionemo/examples/tests/test_data/molecule/diffdock
fi

if ! [ -z "$1" ]
  then
    DATA_PATH=$1
  else
    echo Data will be extracted to $DATA_PATH. You can change location by providing argument to \
    this script:  download_data_sample.sh \<data_path\>
fi

ngc registry resource download-version nvidian/clara-lifesciences/diffdock:processed_sample
tar -xvf diffdock_vprocessed_sample/diffdock_processsed_sample.tar.gz -C $DATA_PATH
rm -r diffdock_vprocessed_sample/
