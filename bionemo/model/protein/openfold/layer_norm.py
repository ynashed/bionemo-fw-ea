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

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer Normalization module.

    Supplementary '1.11.4 Parameters initialization': Layer normalization.

    Args:
        in_channels: Last dimension of the input tensor.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
    ) -> None:
        super(LayerNorm, self).__init__()
        self.normalized_shape = (in_channels,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input=x,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
