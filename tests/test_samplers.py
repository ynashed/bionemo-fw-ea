# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import pytest
from torch.utils.data.sampler import RandomSampler

from bionemo.data.diffdock.sampler import SizeAwareBatchSampler


@pytest.mark.parametrize('dataset_size', (113, 256))
@pytest.mark.parametrize('max_total_size', (1, 32, 100))
def test_sampler_shuffling(dataset_size, max_total_size):
    dataset = range(dataset_size)
    sampler = RandomSampler(dataset)
    sampler = SizeAwareBatchSampler(
        sampler, max_total_size, np.ones(dataset_size), num_batches=dataset_size // max_total_size
    )
    first_pass = list(sampler)
    second_pass = list(sampler)
    assert len(first_pass) == len(second_pass) == sampler.num_batches, "wrong number of batches"
    assert first_pass != second_pass, "repeated order"
    for sample1, sample2 in zip(first_pass, second_pass):
        assert len(sample1) == len(sample2) == max_total_size, "incorrect batch size"
    for samples in (first_pass, second_pass):
        assert len(set(sum(samples, []))) == sampler.num_batches * max_total_size, "incorrect number of samples"


@pytest.mark.parametrize('dataset_size', (113, 256))
@pytest.mark.parametrize('max_size', (1, 7, 9))
@pytest.mark.parametrize('max_total_size', (1, 32, 100))
def test_sampler_order(dataset_size, max_size, max_total_size):
    order = np.random.permutation(range(dataset_size))
    sizes = np.random.randint(1, max_size + 1, dataset_size)

    sampler = SizeAwareBatchSampler(order, max_total_size, sizes, num_batches=dataset_size)
    expected_order = np.array([i for i in order if sizes[i] <= max_total_size])
    num_unique = len(expected_order)

    num_batches = 0
    sampled_order = []
    for sample in sampler:
        sampled_order += sample
        num_batches += 1
        size = sum(sizes[i] for i in sample)
        assert size >= max_total_size - max_size
        assert size <= max_total_size

    for i in range(0, len(sampled_order), num_unique):
        unique_sample = sampled_order[i : i + num_unique]
        assert len(unique_sample) == len(set(unique_sample)), "unexpected duplicates"
        assert (unique_sample == expected_order[: len(unique_sample)]).all(), "incorrect_order"

    assert num_batches == sampler.num_batches


@pytest.mark.timeout(10)
@pytest.mark.parametrize('dataset_size', (1, 10))
@pytest.mark.parametrize('max_total_size', (17, 153))
def test_too_small_batch_size(dataset_size, max_total_size):
    dataset = range(dataset_size)
    sampler = RandomSampler(dataset)
    sizes = np.random.randint(max_total_size + 1, max_total_size + 100, dataset_size)

    with pytest.raises(RuntimeError) as excinfo:
        SizeAwareBatchSampler(
            sampler, max_total_size, sizes, batch_size=max_total_size, num_batches=dataset_size // max_total_size
        )

    assert "No samples can be generated" in str(excinfo.value)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('dataset_size', (1, 10))
@pytest.mark.parametrize('max_total_size', (17, 153))
def test_too_small_batch_size_callable(dataset_size, max_total_size):
    dataset = range(dataset_size)
    sampler = RandomSampler(dataset)
    sizes = np.random.randint(max_total_size + 1, max_total_size + 100, dataset_size)

    with pytest.raises(RuntimeError) as excinfo:
        sampler = SizeAwareBatchSampler(
            sampler, max_total_size, sizes, batch_size=max_total_size, num_batches=dataset_size // max_total_size
        )
        for _ in sampler:
            pass

    assert "No samples can be generated" in str(excinfo.value)
