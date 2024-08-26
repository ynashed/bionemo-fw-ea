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

from typing import Any, List, Protocol, Type

from megatron.core.transformer import TransformerConfig
from nemo.lightning import io

from bionemo.core.model.config import BionemoModelConfig, BionemoTrainableModelConfig, Loss, Model


class MegatronBioNeMoModelConfig(BionemoModelConfig[Model], TransformerConfig, io.NeedsIOMixin):
    """A ModelConfig class for bionemo that supports usage with Megatron models, for example as NeMo2 requires."""

    model_cls: Type[Model]


class MegatronBioNeMoTrainableModelConfig(MegatronBioNeMoModelConfig[Model], BionemoTrainableModelConfig[Model, Loss]):
    """A TrainableModelConfig class for bionemo that supports usage with Megatron models, for example as NeMo2 requires."""


class IOMixinProto(Protocol):
    """A Protocol for the get/set hparam functions of the IOMixin class from NeMo."""

    def set_hparam(self, attribute: str, value: Any, also_change_value: bool = True) -> None:
        """Set the value of an attribute in the config attached to the class by the IOMixin."""
        ...

    def get_hparam(self, attribute: str) -> Any:
        """Get the value of an attribute in the config attached to the class by the IOMixin."""
        ...


def override_mutate_possibly_extra_mutated_fiddle(
    target_cfg: IOMixinProto, source_cfg: IOMixinProto, maybe_mutated_elements_to_clone: List[str]
) -> None:
    """Override the values of the target config with the values of the source config for the given elements.

    This will modify the tracked init hyper-parameter values, as well as modifying the associated attributes in
        self incase they were modified later by post_init code.

    Args:
        target_cfg: The config to update.
        source_cfg: The config to copy values from.
        maybe_mutated_elements_to_clone: The list of elements to copy from the source config to the target config.

    Returns:
        None, the target config is updated in place.
    """
    for f in maybe_mutated_elements_to_clone:
        # 1. Update the tracked config values
        target_cfg.set_hparam(f, source_cfg.get_hparam(f))
        # 2. Update the lazily untracked values (if the same variable name is used post-init)
        setattr(target_cfg, f, getattr(source_cfg, f))
