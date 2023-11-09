# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch


if "USE_FAST_TP" in os.environ:
    from diffdock import TPCUDA as TensorProduct
    from diffdock.TensorProduct import FullyConnectedTPFunction


class FullyConnectedTP(torch.nn.Module):
    def __init__(self, in_irreps: str, sh_irreps: str, out_irreps: str, dtype: torch.dtype = torch.float32):
        super(FullyConnectedTP, self).__init__()
        self._in_irreps = in_irreps
        self._out_irreps = out_irreps
        self._sh_irreps = sh_irreps
        self._tp = TensorProduct(str(in_irreps), str(sh_irreps), str(out_irreps), dtype=dtype)
        self._dtype = dtype
        self.weight_numel = self._tp.weight_numel

    def forward(self, u, v, weight):
        if self._tp._device.index != u.device.index:
            self._tp = TensorProduct(
                str(self._in_irreps),
                str(self._sh_irreps),
                str(self._out_irreps),
                dtype=self._dtype,
                device=str(u.device),
            )
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return self._tp.forward(u.to(self._dtype), v.to(self._dtype), weight.to(self._dtype))
        else:
            return FullyConnectedTPFunction.apply(
                u.to(self._dtype), v.to(self._dtype), weight.to(self._dtype), self._tp
            )
