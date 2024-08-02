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


import torch

from bionemo.llm.data.collate import collate_fn


def test_collate_fn():
    # Create sample data
    sample1 = {
        "text": torch.tensor([1, 2, 3]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, False]),
        "labels": torch.tensor([7, 8, 9]),
        "loss_mask": torch.tensor([True, False, True]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    sample2 = {
        "text": torch.tensor([10, 11, 12]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, False, True]),
        "labels": torch.tensor([16, 17, 18]),
        "loss_mask": torch.tensor([False, True, False]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    batch = [sample1, sample2]

    # Call the collate_fn
    collated_batch = collate_fn(batch, padding_value=-1)

    # Assert the expected output
    assert torch.all(torch.eq(collated_batch["text"], torch.tensor([[1, 2, 3], [10, 11, 12]])))
    assert torch.all(torch.eq(collated_batch["types"], torch.tensor([[0, 0, 0], [0, 0, 0]])))
    assert torch.all(
        torch.eq(collated_batch["attention_mask"], torch.tensor([[True, True, False], [True, False, True]]))
    )
    assert torch.all(torch.eq(collated_batch["labels"], torch.tensor([[7, 8, 9], [16, 17, 18]])))
    assert torch.all(torch.eq(collated_batch["loss_mask"], torch.tensor([[True, False, True], [False, True, False]])))
    assert torch.all(torch.eq(collated_batch["is_random"], torch.tensor([[0, 0, 0], [0, 0, 0]])))


def test_collate_fn_with_padding():
    # Create sample data
    sample1 = {
        "text": torch.tensor([1, 2, 3]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, False]),
        "labels": torch.tensor([7, 8, 9]),
        "loss_mask": torch.tensor([True, False, True]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    sample2 = {
        "text": torch.tensor([4, 5, 6, 7, 8]),
        "types": torch.zeros((5,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, True, True, True]),
        "labels": torch.tensor([-1, 5, -1, 7, 8]),
        "loss_mask": torch.tensor([False, True, False, True, True]),
        "is_random": torch.zeros((5,), dtype=torch.int64),
    }
    batch = [sample1, sample2]

    # Call the collate_fn
    collated_batch = collate_fn(batch, padding_value=10)

    # Assert the expected output
    assert torch.all(torch.eq(collated_batch["text"], torch.tensor([[1, 2, 3, 10, 10], [4, 5, 6, 7, 8]])))
    assert torch.all(torch.eq(collated_batch["types"], torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))
    assert torch.all(
        torch.eq(
            collated_batch["attention_mask"],
            torch.tensor([[True, True, False, False, False], [True, True, True, True, True]]),
        )
    )
    assert torch.all(torch.eq(collated_batch["labels"], torch.tensor([[7, 8, 9, -1, -1], [-1, 5, -1, 7, 8]])))
    assert torch.all(
        torch.eq(
            collated_batch["loss_mask"],
            torch.tensor([[True, False, True, False, False], [False, True, False, True, True]]),
        )
    )
    assert torch.all(torch.eq(collated_batch["is_random"], torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))


def test_collate_fn_with_max_length_truncates():
    # Create sample data
    sample1 = {
        "text": torch.tensor([1, 2, 3]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, False]),
        "labels": torch.tensor([7, 8, 9]),
        "loss_mask": torch.tensor([True, False, True]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    sample2 = {
        "text": torch.tensor([4, 5, 6, 7, 8]),
        "types": torch.zeros((5,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, True, True, True]),
        "labels": torch.tensor([-1, 5, -1, 7, 8]),
        "loss_mask": torch.tensor([False, True, False, True, True]),
        "is_random": torch.zeros((5,), dtype=torch.int64),
    }
    batch = [sample1, sample2]

    # Call the collate_fn
    collated_batch = collate_fn(batch, padding_value=10, max_length=4)

    # Assert the expected output
    assert torch.all(torch.eq(collated_batch["text"], torch.tensor([[1, 2, 3, 10], [4, 5, 6, 7]])))
    assert torch.all(torch.eq(collated_batch["types"], torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])))
    assert torch.all(
        torch.eq(
            collated_batch["attention_mask"], torch.tensor([[True, True, False, False], [True, True, True, True]])
        )
    )
    assert torch.all(torch.eq(collated_batch["labels"], torch.tensor([[7, 8, 9, -1], [-1, 5, -1, 7]])))
    assert torch.all(
        torch.eq(collated_batch["loss_mask"], torch.tensor([[True, False, True, False], [False, True, False, True]]))
    )
    assert torch.all(torch.eq(collated_batch["is_random"], torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])))


def test_collate_fn_with_max_length_pads_extra():
    # Create sample data
    sample1 = {
        "text": torch.tensor([1, 2, 3]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, False]),
        "labels": torch.tensor([7, 8, 9]),
        "loss_mask": torch.tensor([True, False, True]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    sample2 = {
        "text": torch.tensor([10, 11, 12]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, False, True]),
        "labels": torch.tensor([16, 17, 18]),
        "loss_mask": torch.tensor([False, True, False]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    batch = [sample1, sample2]

    # Call the collate_fn
    collated_batch = collate_fn(batch, padding_value=-1, max_length=5)
    assert torch.all(torch.eq(collated_batch["text"], torch.tensor([[1, 2, 3, -1, -1], [10, 11, 12, -1, -1]])))
    for val in collated_batch.values():
        assert val.size(1) == 5
