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
from e3nn import o3

from bionemo.model.molecule.diffdock.utils.fast_tensor_product import FullyConnectedTP


@pytest.mark.needs_gpu
def test_fast_tp_backward():
    dtype = torch.float32
    device = torch.device("cuda:0")
    sizeBatch = 2

    in_irreps, sh_irreps, out_irreps = (
        str("48x0e+10x1e+10x1o+48x0o"),
        str("1x0e+1x1o+1x2e"),
        str("48x0e+10x1e+10x1o+48x0o"),
    )
    fast_tp = FullyConnectedTP(in_irreps, sh_irreps, out_irreps, dtype, device)
    e3nn_tp = (
        o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False).to(device).to(dtype)
    )

    input_tensor_u = torch.randn(
        [sizeBatch, fast_tp._tp._sizeIrrepsLHS], dtype=dtype, device=device, requires_grad=True
    )
    input_tensor_v = torch.randn(
        [sizeBatch, fast_tp._tp._sizeIrrepsRHS], dtype=dtype, device=device, requires_grad=True
    )
    weights = torch.randn((sizeBatch, fast_tp.weight_numel), dtype=dtype, device=device, requires_grad=True)

    ref_out = e3nn_tp(input_tensor_u, input_tensor_v, weights)
    out = fast_tp(input_tensor_u, input_tensor_v, weights)

    assert torch.allclose(out, ref_out, atol=1.0e-2)
    assert torch.autograd.gradcheck(
        fast_tp.forward, [input_tensor_u, input_tensor_v, weights], eps=1e-3, atol=1e-3, rtol=1e-4
    )
