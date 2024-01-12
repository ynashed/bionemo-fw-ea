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

from bionemo.model.protein.openfold.linear import Linear


class TemplatePairEmbedder(nn.Module):
    """Template Pair Embedder module.

    Embeds the "template_pair_feat" feature.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2, line 9.

    Args:
        tp_dim: Input `template_pair_feat` dimension (channels).
        c_t: Output template representation dimension (channels).

    """

    def __init__(
        self,
        tp_dim: int,
        c_t: int,
    ) -> None:
        super(TemplatePairEmbedder, self).__init__()
        self.tp_dim = tp_dim
        self.c_t = c_t
        self.linear = Linear(tp_dim, c_t, bias=True, init="relu")

    def forward(
        self,
        template_pair_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Template Pair Embedder forward pass.

        Args:
            template_pair_feat: [batch, N_res, N_res, tp_dim]

        Returns:
            template_pair_embedding: [batch, N_res, N_res, c_t]

        """
        return self.linear(template_pair_feat)
