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


class SingleTransition(nn.Module):
    """Single Transition module.

    Supplementary '1.8 Structure module': Algorithm 20, lines 8-9.

    Args:
        c_s: Single representation dimension (channels).
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        c_s: int,
        dropout_rate: float,
    ) -> None:
        super(SingleTransition, self).__init__()
        self.linear_1 = Linear(c_s, c_s, bias=True, init="relu")
        self.linear_2 = Linear(c_s, c_s, bias=True, init="relu")
        self.linear_3 = Linear(c_s, c_s, bias=True, init="final")
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(c_s)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = s + self.linear_3(torch.relu(self.linear_2(torch.relu(self.linear_1(s)))))
        s = self.layer_norm(self.dropout(s))
        return s
