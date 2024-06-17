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

from bionemo.contrib.utils.dtypes import get_autocast_dtype


@pytest.mark.parametrize(
    "precision, expected_dtype",
    [
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
        ("fp32", torch.float32),
        ("bf16-mixed", torch.bfloat16),
        ("fp32-mixed", torch.float32),
    ],
)
def test_get_autocast_dtype(precision: str, expected_dtype: torch.dtype):
    assert get_autocast_dtype(precision) == expected_dtype


def test_unsupported_autocast_dtype():
    with pytest.raises(ValueError):
        get_autocast_dtype("unsupported")
