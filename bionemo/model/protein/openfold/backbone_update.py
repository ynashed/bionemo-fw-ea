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
from bionemo.model.protein.openfold.linear import Linear


class BackboneUpdate(nn.Module):
    """Backbone Update module.

    Supplementary '1.8.3 Backbone update': Algorithm 23.

    Args:
        c_s: Single representation dimension (channels).

    """

    def __init__(self, c_s: int) -> None:
        super(BackboneUpdate, self).__init__()
        self.linear = Linear(c_s, 6, bias=True, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: [optim-hub] re-check condition: can we safely ignore dap / do we take into account hopper
        if inductor.is_enabled_on_hopper():  #  and dap.size() in {2, 8}:
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        return forward_fn(s, self.linear.weight, self.linear.bias)


def _forward_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, w, b)


_forward_jit = torch.compile(_forward_eager)
