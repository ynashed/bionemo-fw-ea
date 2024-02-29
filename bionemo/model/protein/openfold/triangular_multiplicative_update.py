# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

import torch
import torch.nn as nn

from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear
from bionemo.model.protein.openfold.utils.torch_utils import is_autocast_fp16_enabled


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle Multiplicative Update module.

    Supplementary '1.6.5 Triangular multiplicative update': Algorithms 11 and 12.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        tmu_type: "outgoing" or "incoming"

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        tmu_type: Literal['outgoing', 'incoming'],
    ) -> None:
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        if tmu_type == "outgoing":
            self._is_outgoing = True
        elif tmu_type == "incoming":
            self._is_outgoing = False
        else:
            raise ValueError(f"Invalid TMU Type '{tmu_type}', must be one of 'outgoing' or 'incoming'")

        self.linear_a_p = Linear(c_z, c_hidden, bias=True, init="default")
        self.linear_a_g = Linear(c_z, c_hidden, bias=True, init="gating")
        self.linear_b_p = Linear(c_z, c_hidden, bias=True, init="default")
        self.linear_b_g = Linear(c_z, c_hidden, bias=True, init="gating")
        self.linear_g = Linear(c_z, c_z, bias=True, init="gating")
        self.linear_z = Linear(c_hidden, c_z, bias=True, init="final")
        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Triangle Multiplicative Update forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z_update: [batch, N_res, N_res, c_z] pair representation update

        """
        z = self.layer_norm_in(z)
        # z: [batch, N_res, N_res, c_z]

        mask = mask.unsqueeze(-1)
        # mask: [batch, N_res, N_res, 1]

        a = torch.sigmoid(self.linear_a_g(z)) * mask
        a = a * self.linear_a_p(z)
        # a: [batch, N_res, N_res, c_hidden]

        b = torch.sigmoid(self.linear_b_g(z)) * mask
        b = b * self.linear_b_p(z)
        # b: [batch, N_res, N_res, c_hidden]

        if is_autocast_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)
        # x: [batch, N_res, N_res, c_hidden]

        del a, b

        x = self.layer_norm_out(x)
        # x: [batch, N_res, N_res, c_hidden]

        x = self.linear_z(x)
        # x: [batch, N_res, N_res, c_z]

        g = torch.sigmoid(self.linear_g(z))
        # g: [batch, N_res, N_res, c_z]

        x = x * g
        # x: [batch, N_res, N_res, c_z]

        return x

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_outgoing:
            a = a.movedim(a.ndim - 1, a.ndim - 3)
            b = b.swapdims(b.ndim - 1, b.ndim - 3)
        else:
            a = a.swapdims(a.ndim - 1, a.ndim - 3)
            b = b.movedim(b.ndim - 1, b.ndim - 3)

        p = torch.matmul(a, b)

        return p.movedim(p.ndim - 3, p.ndim - 1)


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """Triangle Multiplication Outgoing module.

    Supplementary '1.6.5 Triangular multiplicative update':
    Algorithm 11 Triangular multiplicative update using "outgoing" edges.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
    ) -> None:
        super(TriangleMultiplicationOutgoing, self).__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            tmu_type="outgoing",
        )


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """Triangle Multiplication Incoming module.

    Supplementary '1.6.5 Triangular multiplicative update':
    Algorithm 12 Triangular multiplicative update using "incoming" edges.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
    ) -> None:
        super(TriangleMultiplicationIncoming, self).__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            tmu_type="incoming",
        )
