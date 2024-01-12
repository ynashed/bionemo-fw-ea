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


class TemplateAngleEmbedder(nn.Module):
    """Template Angle Embedder module.

    Embeds the "template_angle_feat" feature.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2, line 7.

    Args:
        ta_dim: Input `template_angle_feat` dimension (channels).
        c_m: Output MSA representation dimension (channels).

    """

    def __init__(
        self,
        ta_dim: int,
        c_m: int,
    ) -> None:
        super(TemplateAngleEmbedder, self).__init__()
        self.linear_1 = Linear(ta_dim, c_m, bias=True, init="relu")
        self.linear_2 = Linear(c_m, c_m, bias=True, init="relu")

    def forward(
        self,
        template_angle_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Template Angle Embedder forward pass.

        Args:
            template_angle_feat: [batch, N_templ, N_res, ta_dim]

        Returns:
            template_angle_embedding: [batch, N_templ, N_res, c_m]

        """
        return self.linear_2(torch.relu(self.linear_1(template_angle_feat)))
