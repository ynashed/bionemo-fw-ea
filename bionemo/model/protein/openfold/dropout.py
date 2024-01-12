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

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    """Dropout module.

    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.

    Supplementary '1.11.6 Dropout details'.

    Args:
        p: Dropout rate (probability of an element to be zeroed).
        share_dim: Dimension(s) along which the dropout mask is shared.
        inplace: If set to `True`, will do this operation in-place.

    """

    def __init__(
        self,
        p: float,
        share_dim: Union[int, Tuple[int, ...]] = (),
        inplace: bool = False,
    ) -> None:
        super(Dropout, self).__init__()
        assert 0.0 <= p <= 1.0
        self.p = p
        if isinstance(share_dim, int):
            share_dim = (share_dim,)
        else:
            assert isinstance(share_dim, tuple)
        self.share_dim = share_dim
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        for d in self.share_dim:
            shape[d] = 1
        mask = x.new_ones(shape)
        mask = F.dropout(
            input=mask,
            p=self.p,
            training=self.training,
            inplace=self.inplace,
        )
        x *= mask
        return x


class DropoutRowwise(Dropout):
    """Dropout Rowwise module."""

    def __init__(self, p: float) -> None:
        super(DropoutRowwise, self).__init__(p=p, share_dim=-3)


class DropoutColumnwise(Dropout):
    """Dropout Columnwise module."""

    def __init__(self, p: float) -> None:
        super(DropoutColumnwise, self).__init__(p=p, share_dim=-2)
