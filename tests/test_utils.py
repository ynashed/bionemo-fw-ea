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

import os
import shutil
import tempfile
from nemo_chem.data.utils import create_dataset

class MockConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def _get_node_flag(self, name):
        return None

    def _set_flag(self, key, value):
        return None

def _setup_test_data():
    test_data_path = os.path.join(os.path.dirname(__file__), 'data', 'train')
    tmp_directory = tempfile.TemporaryDirectory()
    files_to_copy = ['x000.csv', 'x001.csv']
    for f in files_to_copy:
        shutil.copy(f'{test_data_path}/{f}', f'{tmp_directory.name}/{f}')
    return tmp_directory, files_to_copy

def test_csv_dataset_construction():
    cfg = MockConfig({
        'header_lines': 1,
        'newline_int': 10,
        'sort_dataset_paths': True,
        'data_col': 1,
        'data_sep': ',',
    })
    test_directory, _ = _setup_test_data()
    dataset_paths = f'{test_directory.name}/(x000,x001)'
    dataset = create_dataset(
        cfg,
        num_samples=None,
        filepath=dataset_paths,
        metadata_path=None,
        dataset_format="csv",
    )
    assert len(dataset) == 400
    assert dataset[0] == 'CCN1CCN(c2ccc(-c3nc(CCN)no3)cc2F)CC1'
    assert dataset[102] == \
        'Cc1c(C(=O)N2CC[C@@H](C(F)(F)F)[C@@H](CN)C2)cnn1C1CCCC1'
    assert dataset[200] == \
        'FC(F)Oc1ccc([C@H](NCc2cnc3ccccn23)C(F)(F)F)cc1'
