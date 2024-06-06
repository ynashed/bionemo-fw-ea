# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pytest

from bionemo.data.diffdock.webdataset_utils import SizeAwareBatching


@dataclass
class Sample:
    """sample or batch of samples for this unit test"""

    name: Union[int, List[int]]
    size: Union[float, int, List[float], List[int]]


def collate_fn(batch):
    names = [data.name for data in batch]
    sizes = [data.size for data in batch]
    return Sample(name=names, size=sizes)


@pytest.mark.parametrize('dataset_size', (113, 256))
@pytest.mark.parametrize('max_size', (1, 7, 9))
@pytest.mark.parametrize('max_total_size', (1, 32, 100))
def test_size_aware_batching_order(dataset_size, max_size, max_total_size):
    order = np.random.permutation(range(dataset_size))
    sizes = np.random.randint(1, max_size + 1, dataset_size)

    expected_order = np.array([i for i in order if sizes[i] <= max_total_size])
    num_unique = len(expected_order)

    dataset = [Sample(name=idx, size=sizes[idx]) for idx in order]

    size_aware_batching = SizeAwareBatching(max_total_size, lambda x: x.size, collate_fn)

    num_batches = 0
    sampled_order = []
    for sample in size_aware_batching(dataset):
        sampled_order += sample.name
        num_batches += 1
        size = sum(sizes[i] for i in sample.name)
        assert size >= max_total_size - max_size
        assert size <= max_total_size

    for i in range(0, len(sampled_order), num_unique):
        unique_sample = sampled_order[i : i + num_unique]
        assert len(unique_sample) == len(set(unique_sample)), "unexpected duplicates"
        assert (unique_sample == expected_order[: len(unique_sample)]).all(), "incorrect_order"


@pytest.mark.parametrize('dataset_size', (1, 10))
@pytest.mark.parametrize('max_total_size', (17, 153))
def test_too_small_batch_size(dataset_size, max_total_size):
    sizes = np.random.randint(max_total_size + 1, max_total_size + 100, dataset_size)
    dataset = [Sample(name=idx, size=sizes[idx]) for idx in range(dataset_size)]

    size_aware_batching = SizeAwareBatching(max_total_size, lambda x: x.size, collate_fn)
    assert len(list(size_aware_batching(dataset))) == 0
