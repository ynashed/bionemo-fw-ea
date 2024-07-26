# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from bionemo.contrib.data.scdl.api.single_cell_row_dataset import SingleCellRowDatasetCore
from bionemo.contrib.data.scdl.index.feature_index import RowFeatureIndex
from bionemo.contrib.data.scdl.io.sc_mmap_dataset import Mode, SC_MMAP_Dataset
from bionemo.contrib.data.scdl.util.string_enum import StringEnum
from bionemo.contrib.data.scdl.util.task_queue import AsyncWorkQueue


__all__: Sequence[str] = (
    "ArrNames",
    "SingleCellCollection",
)

logger = logging.getLogger("sc_collection")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s")


def _create_scmmap_from_h5ad(
    datapath: str, fname: Optional[str] = None, base_dir: Optional[str] = None
) -> SC_MMAP_Dataset:
    """
    Creates and SC_MAP_Dataset. Loads the h5ad file from datapath. The SC_MMAP_Dataset is loaded from datapath,
    base_dir + datapath, or fname Raises an error if both the fnmae and base_dir are sepcified."""
    if base_dir is not None and fname is not None:
        raise ValueError("Only one of 'base_dir' or 'fname' should be set to a value. The other must be None.")
    scmm_path = Path(datapath).name.rsplit(".")[0]
    if base_dir is not None:
        scmm_path = base_dir + "/" + scmm_path
    if fname is not None:
        scmm_path = fname
    obj = SC_MMAP_Dataset(scmm_path)
    obj.load_h5ad(datapath)
    return obj


class ArrNames(StringEnum):
    VERSION = "version.json"
    METADATA = "metadata.json"
    FEATURES = "features.idx"


class SingleCellCollection(SingleCellRowDatasetCore):
    """
    Implements a way of dealing with collections of single-cell datasets,
    e.g., multiple AnnData files.
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self._version: str = "0.0.1"
        # Metadata and attributes
        self.metadata: Dict = {}
        self._feature_index: RowFeatureIndex = RowFeatureIndex()
        # List of SCMMAP object directory paths.
        self.mmap_paths = []
        # dictionary to hold each SCMMAP object.
        self.fname_to_mmap = {}

        # None: unset
        # False: not ragged; all scmmaps have same column dim
        # True: ragged; scmmap column dims vary
        self._is_ragged: Optional[bool] = None

        if not os.path.exists(self.data_path):
            Path(self.data_path).mkdir(parents=True, exist_ok=True)

        # Write the version
        if not os.path.exists(f"{self.data_path}/{ArrNames.VERSION}"):
            with open(f"{self.data_path}/{ArrNames.VERSION}", "w") as vfi:
                json.dump(self.version(), vfi)

    def __len__(self) -> int:
        return self.n_obs()

    def version(self) -> str:
        return self._version

    def load_h5ad(self, h5ad_path: str) -> None:
        mmap_path = str((Path(self.data_path) / Path(h5ad_path).name.rsplit(".")[0]).absolute())
        self.fname_to_mmap[mmap_path] = _create_scmmap_from_h5ad(h5ad_path, fname=None, base_dir=self.data_path)
        self.mmap_paths.append(mmap_path)
        self._feature_index.concat(self.fname_to_mmap[mmap_path]._feature_index)

    def load_h5ad_multi(self, fs_path: str, max_workers: int = 5, use_processes: bool = False) -> None:
        """
        Loads one or more h5ad files.
        """
        fs_path = Path(fs_path)
        ann_data_paths = sorted(fs_path.rglob("*.h5ad"))

        self.mmap_paths = [
            str((Path(self.data_path) / Path(datapath).name.rsplit(".")[0]).absolute()) for datapath in ann_data_paths
        ]

        queue = AsyncWorkQueue(max_workers=max_workers, use_processes=use_processes)
        for ann in ann_data_paths:
            queue.submit_task(_create_scmmap_from_h5ad, ann, fname=None, base_dir=self.data_path)
        queue.wait()
        mmaps = queue.get_task_results()

        for i in range(0, len(self.mmap_paths)):
            self.fname_to_mmap[self.mmap_paths[i]] = mmaps[i]
            self._feature_index.concat(self.fname_to_mmap[self.mmap_paths[i]]._feature_index)

    def num_nonzeros(self) -> int:
        return sum([self.fname_to_mmap[name].num_nonzeros() for name in self.mmap_paths])

    def n_values(self) -> int:
        return sum([self.fname_to_mmap[p].n_values() for p in self.mmap_paths])

    def n_obs(self) -> int:
        return sum([self.fname_to_mmap[p].n_obs() for p in self.mmap_paths])

    def n_vars(self) -> int:
        """
        If ragged, returns a list of variable lengths
        if not ragged, returns a scalar.
        """
        assert len(self.fname_to_mmap) == len(self._feature_index._feature_arr)
        if len(self._feature_index) == 0:
            return 0
        num_vars = self._feature_index.column_dims()
        if len(num_vars) == 1:
            return num_vars[0]
        return num_vars

    def shape(self) -> Tuple[int, int]:
        """
        If ragged, returns a list of shape tuples
        if not ragged, returns a single shape tuple n_obs x n_var
        """

        return self.n_obs(), self.n_vars()

    def sparsity(self) -> float:
        """
        Return the sparsity of the underlying data within the range [0, 1.0].
        """
        return (float(self.n_values()) - float(self.num_nonzeros())) / float(self.n_values())

    def identical_vars(
        self, other_dataset: SC_MMAP_Dataset, cols_to_compare: Optional[List[str]] = None, check_order: bool = True
    ) -> bool:
        if cols_to_compare is None:
            cols_to_compare = ["feature_name"]
        for path in self.mmap_paths:
            logger.info(f"{path}: {self.fname_to_mmap[path].compare_features(other_dataset)}")
        return all(
            self.fname_to_mmap[path].compare_features(other_dataset, cols_to_compare, check_order)
            for path in self.mmap_paths
        )

    def is_column_validate_component_datasets(
        self, cols_to_compare: Optional[List[str]] = None, check_order: bool = True
    ) -> bool:
        if cols_to_compare is None:
            cols_to_compare = ["feature_name"]
        for i in self.mmap_paths:
            for j in self.mmap_paths:
                ds_i = self.fname_to_mmap[i]
                ds_j = self.fname_to_mmap[j]
                if not ds_i.compare_features(ds_j, cols_to_compare=cols_to_compare, check_order=check_order):
                    return False
        return True

    def flatten(
        self,
        output_path: str,
        destroy_on_copy: bool = False,
    ) -> None:
        """
        Flattens the collection into a single SCMMAP.
        """
        output = SC_MMAP_Dataset(
            output_path, n_obs=self.n_obs(), n_values=self.num_nonzeros(), mode=Mode.CREATE_APPEND
        )

        # TODO:
        # implement the following two arguments:
        #   cols_to_compare: Optional[List[str]] = None, with ["feature_name"] as default
        #   check_column_order: bool = True,
        # Then implement the following check, but with a valid condition and handling.
        # if check_column_compatibility or check_column_compatibility or feature_join:
        #     raise NotImplementedError()
        mmaps = [self.fname_to_mmap[path] for path in self.mmap_paths]
        output.concat(mmaps)

        # Hit save!
        output.save()

        if destroy_on_copy:
            shutil.rmtree(self.data_path)
