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

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple, Union

from torch.utils.data import Dataset


__all__: Sequence[str] = (
    "SingleCellRowDataset",
    "DurableData",
    "SingleCellRowDatasetCore",
)


class DurableData(ABC):
    @abstractmethod
    def load(self, data_path: str) -> None:
        """
        Loads the data from datapath.
        Calls to __len__ and __getitem__ Must be valid after a call to this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, data_path: str) -> None:
        """
        Saves the class to an archive at datapath.
        """
        raise NotImplementedError()


class SingleCellRowDatasetCore(ABC):
    @abstractmethod
    def version(self) -> str:
        """
        Returns a version number (following <major>.<minor>.<point> convention).
        """
        raise NotImplementedError()

    @abstractmethod
    def load_h5ad(self, h5ad_path: str) -> None:
        """
        Loads an H5AD file and converts it into the backing representation
        used by the subclass. Calls to __len__ and __getitem__ Must be valid after a call to this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def num_nonzeros(self) -> int:
        """
        Return the number of non-zero values in the data.
        """
        raise NotImplementedError()

    @abstractmethod
    def n_values(self) -> int:
        """
        Return the total number of values in the data.
        """
        raise NotImplementedError()

    @abstractmethod
    def n_obs(self) -> int:
        """
        Return the number of observations (rows) in the data.
        """
        raise NotImplementedError()

    @abstractmethod
    def n_vars(self) -> Union[int, List[int]]:
        """
        Return the number of variables (columns) in the data.
        If a dataset is ragged (i.e., different rows have different numbers of features),
        the returned type is a list of integers.
        """
        raise NotImplementedError()

    @abstractmethod
    def shape(self) -> Union[Tuple[int, int], Tuple[int, List[int]]]:
        """
        Returns the shape of the object, which may be ragged.
        """
        raise NotImplementedError()

    def sparsity(self) -> float:
        """
        Return the sparsity of the underlying data within the range [0, 1.0].
        """
        return (float(self.n_values()) - float(self.num_nonzeros())) / float(self.n_values())


class SingleCellRowDataset(SingleCellRowDatasetCore, DurableData, Dataset, ABC):
    pass
