# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
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


from abc import ABC
from typing import Generic, Sequence, TypeVar

import torch
import torch.distributed
from megatron.core.transformer.module import MegatronModule
from nemo.lightning.megatron_parallel import MegatronLossReduction


__all__: Sequence[str] = (
    "Model",
    "Loss",
)

ModelOutput = TypeVar("ModelOutput", torch.Tensor, list[torch.Tensor], tuple[torch.Tensor], dict[str, torch.Tensor])


class BionemoMegatronModule(MegatronModule, Generic[ModelOutput], ABC):
    pass


Model = TypeVar("Model", bound=BionemoMegatronModule)
Loss = TypeVar("Loss", bound=MegatronLossReduction)
