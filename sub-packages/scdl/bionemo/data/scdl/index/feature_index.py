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
from typing import List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd


__all__: Sequence[str] = ("RowFeatureIndex",)


Self = TypeVar("Self", bound="RowFeatureIndex")


class RowFeatureIndex:
    """Maintains a mapping between a given an row in an scmmemap or sccolection and the features
    associated with that row. (ragged array problem). Preserves the columns for a given row."""

    def __init__(self) -> None:
        # Pointers that deliniate which entries correspond to a given row.
        # for examples if the array is [-1, 200, 201], rows 0 to 199 are assumed to correspond
        # to _feature_arr[0] and 200 corresponds to _feature_arr[1]
        self._cumulative_sum_index: np.array = np.array([-1])
        # List of feature arrays
        self._feature_arr: List[pd.DataFrame] = []
        # Optional labels for the entries
        self._labels: List[str] = []
        # Marks whether entries have been added and the dataset can be looked up
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

    def _update_index(self, n_obs: int, features: pd.DataFrame, label: Optional[str] = None) -> None:
        """
        Updates the index by inserting the dataframe into the feature array
        and adding a new span to the row lookup index.
        """
        csum = max(self._cumulative_sum_index[-1], 0)
        self._cumulative_sum_index = np.append(self._cumulative_sum_index, csum + n_obs)
        self._feature_arr.append(features)
        self._labels.append(label)
        self._index_ready = True

    def append_features(self, n_obs: int, features: pd.DataFrame, label: Optional[str] = None) -> None:
        """Append a new feature into the dataframe."""
        return self._update_index(n_obs=n_obs, features=features, label=label)

    def lookup(self, row: int, select_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
        """Find the features at a given row. It is assumed that the row is non-zero.
        _cumulative_sum_index contains pointers to which rows correspond to given dataframes.
         To obtain a specific row, we determine where it is located in _cumulative_sum_index and then look
        up that dataframe in _feature_arr"""
        if row < 0:
            raise ValueError(f"Row index {row} is not valid. It must be non-negative)")
        if not self._index_ready:
            raise IndexError("There are no dataframes to lookup.")

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

        # Retrieve the features for the identified value.
        features = self._feature_arr[d_id]

        # If specific features are to be selected, filter the features.
        if select_features is not None:
            features = features[select_features]

        # Return the features for the identified range.
        return features, self._labels[d_id]

    def n_vars_at_row(self, row: int) -> int:
        """Return number of variables in a given row having the given feature name."""
        feats, _ = self.lookup(row=row)
        return len(feats)

    def column_dims(self, vars_id: Optional[str] = None) -> List[int]:
        """Return the number of columns in all rows or for given features."""
        if vars_id is None:
            # Just take the total dim of the DataFrame(s)
            return [len(feats.iloc[:, 0]) for feats in self._feature_arr]
        else:
            return [len(feats[vars_id]) for feats in self._feature_arr]

    def n_values(self) -> int:
        """Get the total number of values in the array. For each row, the length of entries
        in the corresponding dataframe is counted.
        """
        if len(self._feature_arr) == 0:
            return 0
        rows = [
            self._cumulative_sum_index[i] - max(self._cumulative_sum_index[i - 1], 0)
            for i in range(1, len(self._cumulative_sum_index))
        ]

        vals = [n_rows * len(self._feature_arr[i]) for i, n_rows in enumerate(rows)]
        return vals

    def n_rows(self) -> int:
        return int(max(self._cumulative_sum_index[-1], 0))

    def concat(self, other_row_index: Self, fail_on_empty_index: bool = True) -> Self:
        """
        Concatenates the other FeatureIndex to this one, returning
        the new, updated index.

        Warning: modifies this index in-place.
        """
        assert isinstance(other_row_index, type(self))
        if fail_on_empty_index and not len(other_row_index._feature_arr) > 0:
            raise ValueError("Error: Cannot append empty FeatureIndex.")
        for i, feats in enumerate(list(other_row_index._feature_arr)):
            c_span = other_row_index._cumulative_sum_index[i + 1]
            label = other_row_index._labels[i]
            self._update_index(c_span, feats, label)

        return self
