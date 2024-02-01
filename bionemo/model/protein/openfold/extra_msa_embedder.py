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


class ExtraMSAEmbedder(nn.Module):
    """Extra MSA Embedder module.

    Embeds the "extra_msa_feat" feature.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2, line 15.

    Args:
        emsa_dim: Input `extra_msa_feat` dimension (channels).
        c_e: Output extra MSA representation dimension (channels).

    """

    def __init__(
        self,
        emsa_dim: int,
        c_e: int,
    ) -> None:
        super(ExtraMSAEmbedder, self).__init__()
        self.linear = Linear(emsa_dim, c_e, bias=True, init="default")

    def forward(
        self,
        extra_msa_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Extra MSA Embedder forward pass.

        Args:
            extra_msa_feat: [batch, N_extra_seq, N_res, emsa_dim]

        Returns:
            extra_msa_embedding: [batch, N_extra_seq, N_res, c_e]

        """
        # TODO: [optim-hub] re-check condition: can we safely ignore dap / do we take into account hopper
        if inductor.is_enabled_on_hopper():  # and dap.size() == 2:
            forward_fn = _forward_jit
        elif inductor.is_enabled_on_ampere_and_autograd_off():
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        return forward_fn(
            extra_msa_feat,
            self.linear.weight,
            self.linear.bias,
        )


def _forward_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, w, b)


_forward_jit = torch.compile(_forward_eager)
