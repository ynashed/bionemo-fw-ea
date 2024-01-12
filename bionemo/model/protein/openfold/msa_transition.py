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

from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear


class MSATransition(nn.Module):
    """MSA Transition module.

    Supplementary '1.6.3 MSA transition': Algorithm 9.

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
        n: `c_m` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_m: int,
        n: int,
    ) -> None:
        super(MSATransition, self).__init__()
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, n * c_m, bias=True, init="relu")
        self.linear_2 = Linear(n * c_m, c_m, bias=True, init="final")

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Transition forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            m_update: [batch, N_seq, N_res, c_m] MSA representation update

        """
        # DeepMind forgets to apply the MSA mask here.
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = torch.relu(m)
        m = self.linear_2(m)
        return m
