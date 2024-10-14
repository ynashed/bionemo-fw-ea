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


import math
from dataclasses import dataclass
from typing import Literal, Sequence, Type, TypeVar

import torch.nn.functional as F

from bionemo.esm2.model.model import ESM2Model, ESM2GenericConfig
from bionemo.llm.utils import iomixin_utils as iom


__all__: Sequence[str] = (
    "AMPLIFYConfig",
    "AMPLIFYModel",
)


class AMPLIFYModel(ESM2Model):
    """AMPLIFY protein language model."""
    pass
        

AMPLIFYModelT = TypeVar("AMPLIFYModelT", bound=AMPLIFYModel)

@dataclass
class AMPLIFYConfig(ESM2GenericConfig, iom.IOMixinWithGettersSetters):
    """Configuration class for AMPLIFY model. """

    # When overriding fields in a dataclass _always_ declare types: https://github.com/python/cpython/issues/123269
    model_cls: Type[AMPLIFYModel] = AMPLIFYModel
    num_layers: int = 32  # 350M, 24 for 120M
    hidden_size: int = 960  # 350M, 640 for 120M
    num_attention_heads: int = 15 # 350M, 10 for 120M
    ffn_hidden_size: int = 3840  # Transformer FFN hidden size. Usually 4 * hidden_size.
    hidden_dropout: float = 0  # AMPLIFY removes dropout from hidden layers and attention
    attention_dropout: float = 0.0  # AMPLIFY does not use attention dropout
    layernorm_epsilon: float = 1.0e-5
    activation_func: str = F.silu  # AMPLIFY MLP
    # TODO: Add support for RMSNorm
    init_method_std: float = 0.02
    add_bias_linear: bool = False # AMPLIFY does not use bias in linear layers
    normalization: "RMSNorm"    # AMPLIFY uses RMSNorm instead of LayerNorm
    position_embedding_type: Literal["learned_absolute", "rope"] = (
        "rope"  # AMPLIFY uses relative positional encoding 'ROPE' to extrapolate to longer sequences unseen during training
    )
    rotary_base: int = 10000
    rotary_percent: float = 1.0

    make_vocab_size_divisible_by: int = 1
