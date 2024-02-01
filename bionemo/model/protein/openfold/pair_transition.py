# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F

import bionemo.model.protein.openfold.inductor as inductor
from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear


class PairTransition(nn.Module):
    """Pair Transition module.

    Supplementary '1.6.7 Transition in the pair stack': Algorithm 15.

    Args:
        c_z: Pair or template representation dimension (channels).
        n: `c_z` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        n: int,
    ) -> None:
        super(PairTransition, self).__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear_1 = Linear(c_z, n * c_z, bias=True, init="relu")
        self.linear_2 = Linear(n * c_z, c_z, bias=True, init="final")

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pair Transition forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        # DeepMind forgets to apply the pair mask here.
        input_z = z

        z = self.layer_norm(z)

        # make inductor happy - but why? what is the problem with original shape?
        original_shape = z.shape
        z = z.view(-1, z.shape[-1])

        if inductor.is_enabled_on_ampere():
            linear_relu_fn = _linear_relu_jit
        if inductor.is_enabled_on_hopper():
            linear_relu_fn = _linear_relu_jit
        else:
            linear_relu_fn = _linear_relu_eager
        z = linear_relu_fn(z, self.linear_1.weight, self.linear_1.bias)

        # TODO: [optim-hub] This can be jitted if dap is incorporated and dap size >= 2
        z = _linear_view_add_eager(z, self.linear_2.weight, self.linear_2.bias, input_z)

        z = z.view(original_shape)
        return z


def _linear_relu_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.relu(F.linear(x, w, b))


_linear_relu_jit = torch.compile(_linear_relu_eager)


def _linear_view_add_eager(
    z: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    z = F.linear(z, w, b)
    z = z.view(out.shape)
    z = out + z
    return z
