# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
This module contains implementations of particular variants of FullyConnectedTensorProduct
with improved performance when compared to the original e3nn implementation.
Note that code repetitions and lack of flexibility is intentional - it makes scipted modules faster

Available variants:
  * TPFirst:   {ns}x0e                          1x0e+1x1o+1x2e  ->  {ns}x0e+{nv}x1o
  * TPSecond:  {ns}x0e+{nv}x1o                  1x0e+1x1o+1x2e  ->  {ns}x0e+{nv}x1o+{nv}x1e
  * TPThird:   {ns}x0e+{nv}x1o+{nv}x1e          1x0e+1x1o+1x2e  ->  {ns}x0e+{nv}x1o+{nv}x1e+{ns}x0o
  * TPForth:   {ns}x0e+{nv}x1o+{nv}x1e+{ns}x0o  1x0e+1x1o+1x2e  ->  {ns}x0e+{nv}x1o+{nv}x1e+{ns}x0o
"""


import math

import torch
from e3nn import o3
from torch import nn


class TPFirst(nn.Module):
    def __init__(self, ns: int = 48, nv: int = 10) -> None:
        super().__init__()
        self.ns = ns
        self.nv = nv
        self._w_shapes = ((ns, ns), (ns, nv))
        self._w_slices = tuple(math.prod(w_shape) for w_shape in self._w_shapes)
        self.weight_numel = sum(self._w_slices)
        self.norm_coeff_0e = math.sqrt(1 / self.ns)
        self.norm_coeff_1o = math.sqrt(3 / self.ns)

    def forward(self, x1, x2, weights):
        output_shape = x1.shape[:-1] + (self.ns * 1 + self.nv * 3,)
        if 0 in output_shape:
            return torch.zeros(output_shape, dtype=x1.dtype, device=x1.device)

        x1_0e = x1.reshape(-1, self.ns)
        x2_0e, x2_1o, x2_2e = torch.split(x2, [1, 3, 5], -1)
        batch_numel = x1_0e.shape[0]

        weights = torch.split(weights, self._w_slices, 1)
        w00, w01 = [
            weights[i].reshape((-1, self._w_shapes[i][0], self._w_shapes[i][1])) for i in range(len(self._w_shapes))
        ]

        result_0e = self.norm_coeff_0e * torch.einsum("zu,zv,zuw->zw", x1_0e, x2_0e, w00)
        result_1o = self.norm_coeff_1o / math.sqrt(3) * torch.einsum("zu,zj,zuw->zwj", x1_0e, x2_1o, w01)

        output = torch.cat([r.reshape(batch_numel, -1) for r in (result_0e, result_1o)], dim=1).reshape(output_shape)
        return output


class TPSecond(nn.Module):
    def __init__(self, ns: int = 48, nv: int = 10) -> None:
        super().__init__()
        self.ns = ns
        self.nv = nv
        self._w_shapes = (
            (ns, ns),
            (ns, nv),
            (nv, nv),
            (nv, ns),
            (nv, nv),
            (nv, nv),
        )
        self._w_slices = tuple(math.prod(w_shape) for w_shape in self._w_shapes)
        self.weight_numel = sum(self._w_slices)

        self.register_buffer('w3j111', o3.wigner_3j(1, 1, 1), persistent=False)
        self.register_buffer('w3j121', o3.wigner_3j(1, 2, 1), persistent=False)

        self.norm_coeff_0e = math.sqrt(1 / (self.ns + self.nv))
        self.norm_coeff_1o = math.sqrt(3 / (self.ns + self.nv + self.nv))
        self.norm_coeff_1e = math.sqrt(3 / self.nv)

    def forward(self, x1, x2, weights):
        output_shape = x1.shape[:-1] + (self.ns * 1 + self.nv * 3 * 2,)
        if 0 in output_shape:
            return torch.zeros(output_shape, dtype=x1.dtype, device=x1.device)

        x1_0e, x1_1o = torch.split(x1, [self.ns, self.nv * 3], -1)
        x1_0e = x1_0e.reshape((-1, self.ns))
        x1_1o = x1_1o.reshape((-1, self.nv, 3))
        batch_numel = x1_0e.shape[0]

        x2_0e, x2_1o, x2_2e = torch.split(x2, [1, 3, 5], -1)

        weights = torch.split(weights, self._w_slices, 1)
        w00, w01, w10, w110, w112, w12 = [
            weights[i].reshape((-1, self._w_shapes[i][0], self._w_shapes[i][1])) for i in range(len(self._w_shapes))
        ]

        result_0e = self.norm_coeff_0e * torch.einsum("zu,zv,zuw->zw", x1_0e, x2_0e, w00)
        result_1o = self.norm_coeff_1o / math.sqrt(3) * torch.einsum("zu,zj,zuw->zwj", x1_0e, x2_1o, w01)

        result_1o = result_1o + self.norm_coeff_1o / math.sqrt(3) * torch.einsum("zui,zv,zuw->zwi", x1_1o, x2_0e, w10)
        result_0e = result_0e + self.norm_coeff_0e / math.sqrt(3) * torch.einsum("zui,zi,zuw->zw", x1_1o, x2_1o, w110)
        result_1e = self.norm_coeff_1e * torch.einsum("zui,zj,ijk,zuw->zwk", x1_1o, x2_1o, self.w3j111, w112)
        result_1o = result_1o + self.norm_coeff_1o * torch.einsum(
            "zui,zj,ijk,zuw->zwk", x1_1o, x2_2e, self.w3j121, w12
        )

        output = torch.cat([r.reshape(batch_numel, -1) for r in (result_0e, result_1o, result_1e)], dim=1).reshape(
            output_shape
        )
        return output


class TPThird(nn.Module):
    def __init__(self, ns: int = 48, nv: int = 10) -> None:
        super().__init__()
        self.ns = ns
        self.nv = nv
        self._w_shapes = (
            (ns, ns),
            (ns, nv),
            (nv, nv),
            (nv, ns),
            (nv, nv),
            (nv, nv),
            (nv, nv),
            (nv, nv),
            (nv, ns),
            (nv, nv),
        )
        self._w_slices = tuple(math.prod(w_shape) for w_shape in self._w_shapes)
        self.weight_numel = sum(self._w_slices)

        self.register_buffer('w3j111', o3.wigner_3j(1, 1, 1), persistent=False)
        self.register_buffer('w3j121', o3.wigner_3j(1, 2, 1), persistent=False)

        self.norm_coeff_0e = math.sqrt(1 / (ns + nv))
        self.norm_coeff_1o = math.sqrt(3 / (ns + 3 * nv))
        self.norm_coeff_1e = math.sqrt(1 / nv)
        self.norm_coeff_0o = math.sqrt(1 / nv)

    def forward(self, x1, x2, weights):
        output_shape = x1.shape[:-1] + (self.ns * 1 + self.nv * 3 * 2 + self.ns * 1,)
        if 0 in output_shape:
            return torch.zeros(output_shape, dtype=x1.dtype, device=x1.device)

        x1_0e, x1_1o, x1_1e = torch.split(x1, [self.ns, self.nv * 3, self.nv * 3], -1)
        x1_0e = x1_0e.reshape((-1, self.ns))
        x1_1o = x1_1o.reshape((-1, self.nv, 3))
        x1_1e = x1_1e.reshape((-1, self.nv, 3))

        batch_numel = x1_0e.shape[0]

        x2_0e, x2_1o, x2_2e = torch.split(x2, [1, 3, 5], -1)

        weights = torch.split(weights, self._w_slices, 1)
        w00, w01, w10, w110, w112, w12, w20, w211, w213, w22 = [
            weights[i].reshape((-1, self._w_shapes[i][0], self._w_shapes[i][1])) for i in range(len(self._w_shapes))
        ]

        result_0e = self.norm_coeff_0e * torch.einsum("zu,zj,zuw->zw", x1_0e, x2_0e, w00)
        result_1o = self.norm_coeff_1o * torch.einsum("zu,zj,zuw->zwj", x1_0e, x2_1o, w01) / math.sqrt(3)

        result_1o = result_1o + self.norm_coeff_1o * torch.einsum("zui,zj,zuw->zwi", x1_1o, x2_0e, w10) / math.sqrt(3)
        result_0e = result_0e + self.norm_coeff_0e * torch.einsum("zui,zi,zuw->zw", x1_1o, x2_1o, w110) / math.sqrt(3)
        result_1e = self.norm_coeff_1e * torch.einsum("zui,zj,ijk,zuw->zwk", x1_1o, x2_1o, self.w3j111, w112)
        result_1o = result_1o + self.norm_coeff_1o * torch.einsum(
            "zui,zj,ijk,zuw->zwk", x1_1o, x2_2e, self.w3j121, w12
        )

        result_1e = result_1e + self.norm_coeff_1e * torch.einsum("zui,zj,zuw->zwi", x1_1e, x2_0e, w20) / math.sqrt(3)
        result_1o = result_1o + self.norm_coeff_1o * torch.einsum(
            "zui,zj,ijk,zuw->zwk", x1_1e, x2_1o, self.w3j111, w211
        )
        result_0o = self.norm_coeff_0o * torch.einsum("zui,zi,zuw->zw", x1_1e, x2_1o, w213) / math.sqrt(3)
        result_1e = result_1e + self.norm_coeff_1e * torch.einsum(
            "zui,zj,ijk,zuw->zwk", x1_1e, x2_2e, self.w3j121, w22
        )

        output = torch.cat(
            [r.reshape(batch_numel, -1) for r in (result_0e, result_1o, result_1e, result_0o)], dim=1
        ).reshape(output_shape)
        return output


class TPForth(nn.Module):
    def __init__(self, ns: int = 48, nv: int = 10) -> None:
        super().__init__()
        self.ns = ns
        self.nv = nv
        self._w_shapes = (
            (ns, ns),
            (ns, nv),
            (nv, nv),
            (nv, ns),
            (nv, nv),
            (nv, nv),
            (nv, nv),
            (nv, nv),
            (nv, ns),
            (nv, nv),
            (ns, ns),
            (ns, nv),
        )
        self._w_slices = tuple(math.prod(w_shape) for w_shape in self._w_shapes)
        self.weight_numel = sum(self._w_slices)

        self.register_buffer('w3j111', o3.wigner_3j(1, 1, 1), persistent=False)
        self.register_buffer('w3j121', o3.wigner_3j(1, 2, 1), persistent=False)
        self.norm_coeff_0e = math.sqrt(1 / (ns + nv))
        self.norm_coeff_1o = math.sqrt(3 / (ns + 3 * nv))
        self.norm_coeff_1e = math.sqrt(3 / (3 * nv + ns))
        self.norm_coeff_0o = math.sqrt(1 / (nv + ns))

    def forward(self, x1, x2, weights):
        output_shape = x1.shape[:-1] + (self.ns * 1 + self.nv * 3 * 2 + self.ns * 1,)
        if 0 in output_shape:
            return torch.zeros(output_shape, dtype=x1.dtype, device=x1.device)

        x1_0e, x1_1o, x1_1e, x1_0o = torch.split(x1, [self.ns, self.nv * 3, self.nv * 3, self.ns], -1)
        x1_0e = x1_0e.reshape((-1, self.ns))
        x1_1o = x1_1o.reshape((-1, self.nv, 3))
        x1_1e = x1_1e.reshape((-1, self.nv, 3))
        x1_0o = x1_0o.reshape((-1, self.ns))

        batch_numel = x1_0e.shape[0]

        x2_0e, x2_1o, x2_2e = torch.split(x2, [1, 3, 5], -1)

        weights = torch.split(weights, self._w_slices, 1)
        w00, w01, w10, w110, w112, w12, w20, w211, w213, w22, w30, w31 = [
            weights[i].reshape((-1, self._w_shapes[i][0], self._w_shapes[i][1])) for i in range(len(self._w_shapes))
        ]

        result_0e = self.norm_coeff_0e * torch.einsum("zu,zj,zuw->zw", x1_0e, x2_0e, w00)
        result_1o = self.norm_coeff_1o * torch.einsum("zu,zj,zuw->zwj", x1_0e, x2_1o, w01) / math.sqrt(3)

        result_1o = result_1o + self.norm_coeff_1o * torch.einsum("zui,zj,zuw->zwi", x1_1o, x2_0e, w10) / math.sqrt(3)
        result_0e = result_0e + self.norm_coeff_0e * torch.einsum("zui,zi,zuw->zw", x1_1o, x2_1o, w110) / math.sqrt(3)
        result_1e = self.norm_coeff_1e * torch.einsum("zui,zj,ijk,zuw->zwk", x1_1o, x2_1o, self.w3j111, w112)
        result_1o = result_1o + self.norm_coeff_1o * torch.einsum(
            "zui,zj,ijk,zuw->zwk", x1_1o, x2_2e, self.w3j121, w12
        )

        result_1e = result_1e + self.norm_coeff_1e * torch.einsum("zui,zj,zuw->zwi", x1_1e, x2_0e, w20) / math.sqrt(3)
        result_1o = result_1o + self.norm_coeff_1o * torch.einsum(
            "zui,zj,ijk,zuw->zwk", x1_1e, x2_1o, self.w3j111, w211
        )
        result_0o = self.norm_coeff_0o * torch.einsum("zui,zi,zuw->zw", x1_1e, x2_1o, w213) / math.sqrt(3)
        result_1e = result_1e + self.norm_coeff_1e * torch.einsum(
            "zui,zj,ijk,zuw->zwk", x1_1e, x2_2e, self.w3j121, w22
        )

        result_0o = result_0o + self.norm_coeff_0o * torch.einsum("zu,zj,zuw->zw", x1_0o, x2_0e, w30)
        result_1e = result_1e + self.norm_coeff_1e * torch.einsum("zu,zj,zuw->zwj", x1_0o, x2_1o, w31) / math.sqrt(3)

        output = torch.cat(
            [r.reshape(batch_numel, -1) for r in (result_0e, result_1o, result_1e, result_0o)], dim=1
        ).reshape(output_shape)
        return output
