# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Optional

import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        dropout: float = 0.25,
        activation_function: Optional[nn.Module] = None,
    ):
        """
        Simple MLP Model for validation on benchmark datasets

        Params
            layer_sizes: List of layer sizes. By default: [256, 128, 1]
            dropout: float
            activation_function: PyTorch activation function. Uses ReLU if not provided
        """
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [256, 128, 1]
        self.linear_layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layer_norm = nn.LayerNorm(layer_sizes[0])
        self.act = nn.ReLU() if activation_function is None else activation_function
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.linear_layers[:-1]:
            x = self.dropout(self.act(layer(x)))

        x = self.linear_layers[-1](x)
        return x
