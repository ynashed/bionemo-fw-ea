# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest
import torch
from torch.utils.data import TensorDataset

from bionemo.data.mapped_dataset import SliceDataset


test_dataset = TensorDataset(torch.arange(24 * 3).reshape(24, 3))


start_end_length = [
    (0, 9, 9),
    (1, 7, 6),
    (3, 3, 0),
    (9, 24, 15),
    (3, -1, 20),
    (-5, -2, 3),
]


@pytest.mark.parametrize('start,end,length', start_end_length)
def test_slice_dataset(start, end, length):
    dataset = SliceDataset(test_dataset, start, end)
    assert len(dataset) == length
