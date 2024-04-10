# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright (c) 2023, NVIDIA CORPORATION.
"""This file tests some of the utility functions that are used during unit tests."""
import torch

from bionemo.utils.tests import (
    list_to_tensor,
)


def test_list_to_tensor_simple_list():
    data = [1, 2, 3, 4, 5]
    tensor = list_to_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == data


def test_list_to_tensor_nested_list():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    tensor = list_to_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.tolist() == data


def test_list_to_tensor_mixed_list():
    data = [1, 2, [3, 4, 5], [6, [7, 8], 9]]
    tensor = list_to_tensor(data)
    assert isinstance(tensor, list)
    assert isinstance(tensor[3], list)


def test_list_to_tensor_non_list_input():
    data = 42
    tensor = list_to_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.item() == data
