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

import pickle
from typing import List, Optional, Sequence, TypeVar, Union

import numpy as np
from pandas import DataFrame

from bionemo.data.scdl.api.single_cell_row_dataset import SingleCellRowDataset


__all__: Sequence[str] = ("RowFeatureIndex",)


Self = TypeVar("Self", bound="RowFeatureIndex")


class RowFeatureIndex:
    def __init__(self) -> None:
        self._cumulative_sum_index: np.array = np.array([-1])
        self._feature_arr: List[DataFrame] = []
        self._labels: List[str] = []
        self._index_ready: bool = False
        self._version: str = "0.0.1"

    def version(self) -> str:
        return self._version

    def save(self, datapath: str) -> None:
        with open(datapath, "wb") as ofi:
            pickle.dump(self, ofi)

    @staticmethod
    def load(datapath: str) -> None:
        with open(datapath, "rb") as ifi:
            obj = pickle.load(ifi)
            return obj

    def __len__(self) -> int:
        return len(self._feature_arr)

    def _update_index(self, n_obs: int, features: DataFrame, label: Optional[str] = None) -> None:
        """
        Updates the index by inserting the dataframe into the feature array
        and adding a new span to the row lookup index.
        """
        arr_list = list(self._cumulative_sum_index)
        csum = max(self._cumulative_sum_index[-1], 0)
        arr_list.append(csum + n_obs)
        self._cumulative_sum_index = np.array(arr_list)
        self._feature_arr.append(features)
        self._labels.append(label)

        self._index_ready = True

    def append_features(self, n_obs: int, features: DataFrame, label: Optional[str] = None) -> None:
        return self._update_index(n_obs=n_obs, features=features, label=label)

    def index(self, dataset: Union[SingleCellRowDataset, List[SingleCellRowDataset]]) -> None:
        if isinstance(dataset, list):
            for i, d in enumerate(dataset):
                assert hasattr(d, "features")
                assert hasattr(d, "n_obs")
                self.concat(dataset.features())
        elif isinstance(dataset, SingleCellRowDataset):
            assert hasattr(dataset, "features")
            assert hasattr(dataset, "n_obs")
            self.concat(dataset.features())
        else:
            raise ValueError(f"Dataset is of unsupported type: {type(dataset)}")

    def lookup(
        self, row: int, select_features: Optional[List[str]] = None, return_label: bool = False
    ) -> Union[List[DataFrame, int], DataFrame]:
        assert row >= 0
        assert self._index_ready
        if row > self._cumulative_sum_index[-1]:
            raise IndexError(
                f"Row index ({row}) was larger than number of rows in FeatureIndex ({self._cumulative_sum_index[-1]})"
            )
        # This line does the following:
        # creates a mask for values where cumulative sum > row
        mask = ~(self._cumulative_sum_index > row)
        # Sum these to get the index of the first range > row
        # Subtract one to get the range containing row.
        d_id = sum(mask) - 1

        # Retrieve the features for the identified range.
        features = self._feature_arr[d_id]

        # If specific features are to be selected, filter the features.
        if select_features is not None:
            features = features[select_features]

        # If the label is also to be returned, return both features and the label for the identified range.
        if return_label:
            return features, self._labels[d_id]

        # Return the features for the identified range.
        return features

    def n_vars_at_row(self, row: int, vars_id: str = "feature_name") -> int:
        feats = self.lookup(row=row)
        assert vars_id in feats.columns
        return len(feats[vars_id])

    def column_dims(self, vars_id: Optional[str] = None) -> List[int]:
        if vars_id is None:
            # Just take the total dim of the DataFrame(s)
            return [len(feats.iloc[:, 0]) for feats in self._feature_arr]
        else:
            return [len(feats[vars_id]) for feats in self._feature_arr]

    def n_values(self) -> Union[List[int], int]:
        if len(self._feature_arr) == 0:
            return 0
        rows = [
            self._cumulative_sum_index[i] - max(self._cumulative_sum_index[i - 1], 0)
            for i in range(1, len(self._cumulative_sum_index))
        ]
        assert len(rows) == len(self._feature_arr)

        vals = [n_rows * len(self._feature_arr[i].iloc[:, 0]) for i, n_rows in enumerate(rows)]
        return vals

    def n_rows(self) -> int:
        return int(max(self._cumulative_sum_index[-1], 0))

    def concat(self, other: Self, fail_on_empty_index: bool = True) -> Self:
        """
        Concatenates the other FeatureIndex to this one, returning
        the new, updated index.

        Warning: modifies this index in-place.
        """
        assert isinstance(other, type(self))
        if fail_on_empty_index and not len(other._feature_arr) > 0:
            raise ValueError("Error: Cannot append empty FeatureIndex.")
        for i, feats in enumerate(list(other._feature_arr)):
            c_span = other._cumulative_sum_index[i + 1]
            label = other._labels[i]
            self._update_index(c_span, feats, label)

        return self
