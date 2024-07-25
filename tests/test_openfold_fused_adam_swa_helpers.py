# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test module to check correct functionality of"""

import torch
from torch import Tensor  # noqa

from bionemo.model.protein.openfold.swa import swap_tensor_values
from bionemo.model.protein.openfold.triton.fused_adam_swa import _adam_math_0, _swa_math_0, kPyTorchAdam


def test_000_swa_math():
    """Check the exponential moving average arithmatic."""
    swa_param_in = torch.tensor([0, 0], dtype=torch.float32)
    param_in = torch.tensor([2, 2], dtype=torch.float32)
    decay_rate_in = 0.5
    n_averaged_in = 1

    expected_swa_params_out = torch.tensor(
        [1.0, 1.0],
        dtype=torch.float32,
    )

    swa_param_out = _swa_math_0(
        param=param_in,
        swa_param=swa_param_in,
        decay_rate=decay_rate_in,
        n_averaged=n_averaged_in,
    )

    torch.testing.assert_allclose(expected_swa_params_out, swa_param_out, rtol=1e-3, atol=1e-06)


def test_010_adam_math():
    param_in = torch.tensor([1, 1], dtype=torch.float32)
    grad_in = torch.tensor([0, 0], dtype=torch.float32)
    moment_in = torch.tensor([0, 0], dtype=torch.float32)
    velocity_in = torch.tensor([0, 0], dtype=torch.float32)

    beta1 = 0.5
    beta2 = 0.5
    beta1_correction = 0.5
    beta2_correction = 0.5

    param_out, moment_out, velocity_out = _adam_math_0(
        param=param_in,
        grad=grad_in,
        moment=moment_in,
        velocity=velocity_in,
        beta1=beta1,
        beta2=beta2,
        beta1_correction=beta1_correction,
        beta2_correction=beta2_correction,
        eps=1e-6,
        lr=0.1,
        weight_decay=0.5,
        adam_math_mode=kPyTorchAdam,
        compilation_mode="",
    )
    assert isinstance(param_out, Tensor)
    assert isinstance(moment_out, Tensor)
    assert isinstance(velocity_out, Tensor)


def test_020():
    zeros_tensor = torch.zeros(size=(3, 3))
    ones_tensor = torch.ones(size=(3, 3))

    swap_tensor_values(zeros_tensor, ones_tensor)

    assert torch.equal(zeros_tensor, torch.ones(size=(3, 3)))
    assert torch.equal(ones_tensor, torch.zeros(size=(3, 3)))
