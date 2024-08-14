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

import importlib.metadata
import json
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bionemo.scdl.api.single_cell_row_dataset import SingleCellRowDatasetCore
from bionemo.scdl.index.row_feature_index import RowFeatureIndex
from bionemo.scdl.io.single_cell_memmap_dataset import Mode, SingleCellMemMapDataset
from bionemo.scdl.util.async_worker_queue import AsyncWorkQueue


__all__: Sequence[str] = (
    "FileNames",
    "SingleCellCollection",
)

logger = logging.getLogger("sc_collection")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s")


def _create_single_cell_memmap_dataset_from_h5ad(h5ad_path: str, base_directory_path: str) -> SingleCellMemMapDataset:
    """The SingleCellMemMapDataset is loaded from h5ad_path.

    The data is stored in the base_data_path directory.

    Args:
        h5ad_path: the path to the dataset
        base_directory_path: the base directory path where the dataset will be stored
    Returns:
        The created SingleCellMemMapDataset
    """
    fname = Path(h5ad_path).stem
    obj = SingleCellMemMapDataset(data_path=Path(base_directory_path) / fname, h5ad_path=h5ad_path)
    return obj


class FileNames(str, Enum):
    """Names of files that are generated in SingleCellCollection."""

    VERSION = "version.json"
    METADATA = "metadata.json"
    FEATURES = "features"


class SingleCellCollection(SingleCellRowDatasetCore):
    """A collection of one or more SingleCellMemMapDatasets.

    SingleCellCollection support most of the functionality of the
    SingleCellDataSet API. An SingleCellCollection can be converted
    to a single SingleCellMemMapDataset. A SingleCellCollection
    enables the use of heterogeneous datasets, such as those composed of many
    AnnData files.

    Attributes:
        _version: The version of the dataset
        data_path: The directory where the colleection of datasets is stored.
        _feature_index: The corresponding RowFeatureIndex where features are
        stored.
        fname_to_mmap:  dictionary to hold each SingleCellMemMapDataset object.
        This maps from the path to the dataset.
        ragged dataset is an dataset of arrays where the arrays have different
        lengths
        False: not ragged; all SingleCellMemMapDataset have same column dimemsion
        True: ragged; scmmap column dimemsions vary
    """

    def __init__(self, data_path: str) -> None:
        """Instantiate the class.

        Args:
            data_path: Where the class will be stored.
        """
        self.data_path: str = data_path
        self._version: str = importlib.metadata.version("bionemo.scdl")
        self.metadata: Dict[str, int] = {}
        self._feature_index: RowFeatureIndex = RowFeatureIndex()
        self.fname_to_mmap: Dict[str, SingleCellMemMapDataset] = {}

        Path(self.data_path).mkdir(parents=True, exist_ok=True)

        # Write the version
        if not os.path.exists(f"{self.data_path}/{FileNames.VERSION}"):
            with open(f"{self.data_path}/{FileNames.VERSION}", "w") as vfi:
                json.dump(self.version(), vfi)

    def version(self) -> str:
        """Returns a version number.

        (following <major>.<minor>.<point> convention).
        """
        return self._version

    def load_h5ad(self, h5ad_path: str) -> None:
        """Loads data from an existing AnnData archive.

        This creates and saves a new backing data structure.
        Then, the location and the data and the dataset are stored.

        Args:
            h5ad_path: the path to AnnData archive
        """
        mmap_path = Path(self.data_path) / Path(h5ad_path).stem
        self.fname_to_mmap[mmap_path] = _create_single_cell_memmap_dataset_from_h5ad(
            h5ad_path=h5ad_path, base_directory_path=self.data_path
        )
        self._feature_index.concat(self.fname_to_mmap[mmap_path]._feature_index)

    def load_h5ad_multi(self, directory_path: str, max_workers: int = 5, use_processes: bool = False) -> None:
        """Loads one or more AnnData files and adds them to the collection.

        Args:
            directory_path: The path to the directory with the AnnData files
            max_workers: the maximal number of workers to use
            use_processes: If True, use ProcessPoolExecutor; otherwise, use
            ThreadPoolExecutor
        """
        directory_path = Path(directory_path)
        ann_data_paths = sorted(directory_path.rglob("*.h5ad"))

        mmap_paths = [Path(self.data_path) / Path(ann_datapath).stem for ann_datapath in ann_data_paths]
        queue = AsyncWorkQueue(max_workers=max_workers, use_processes=use_processes)
        for ann in ann_data_paths:
            queue.submit_task(_create_single_cell_memmap_dataset_from_h5ad, ann, base_directory_path=self.data_path)
        queue.wait()
        mmaps = queue.get_task_results()

        for mmap_path, mmap in zip(mmap_paths, mmaps):
            self.fname_to_mmap[mmap_path] = mmap
            self._feature_index.concat(self.fname_to_mmap[mmap_path]._feature_index)

    def number_nonzero_values(self) -> int:
        """Sum of the number of non zero entries in each dataset."""
        return sum([self.fname_to_mmap[mmap_path].number_nonzero_values() for mmap_path in self.fname_to_mmap])

    def number_of_values(self) -> int:
        """Sum of the number of values in each dataset."""
        return sum([self.fname_to_mmap[mmap_path].number_of_values() for mmap_path in self.fname_to_mmap])

    def number_of_rows(self) -> int:
        """The number of rows in the dataset.

        Returns:
            The number of rows in the dataset
        Raises:
            ValueError if the length of the number of rows in the feature
            index does not correspond to the number of stored rows.
        """
        row_sum_from_datasets = sum(
            [self.fname_to_mmap[mmap_path].number_of_rows() for mmap_path in self.fname_to_mmap]
        )
        if len(self._feature_index) > 0 and self._feature_index.number_of_rows() != row_sum_from_datasets:
            raise ValueError(
                f"""The nuber of rows in the feature index {self._feature_index.number_of_rows()}
                             does not correspond to the number of rows in the datasets {row_sum_from_datasets}"""
            )

        return row_sum_from_datasets

    def number_of_variables(self) -> List[int]:
        """If ragged, returns a list of variable lengths.

        If not ragged, returns a list with one entry. A ragged
        collection is one where the datasets have different lengths.
        """
        if len(self._feature_index) == 0:
            return [0]
        else:
            num_vars = self._feature_index.column_dims()
            return num_vars

    def shape(self) -> Tuple[int, List[int]]:
        """Get the shape of the dataset.

        This is the number of entries by the the length of the feature index
        corresponding to that variable.

        Returns:
            The total number of elements across dataset
            A list containing the number of variables for each entry in the
            RowFeatureIndex.
        """
        return self.number_of_rows(), self.number_of_variables()

    def flatten(
        self,
        output_path: str,
        destroy_on_copy: bool = False,
    ) -> None:
        """Flattens the collection into a single SingleCellMemMapDataset.

        Args:
            output_path: location to store new dataset
            destroy_on_copy: Whether to remove the current data_path
        """
        output = SingleCellMemMapDataset(
            output_path,
            num_elements=self.number_of_rows(),
            num_rows=self.number_nonzero_values(),
            mode=Mode.CREATE_APPEND,
        )

        output.concat(list(self.fname_to_mmap.values()))

        # Hit save!
        output.save()

        if destroy_on_copy:
            shutil.rmtree(self.data_path)
