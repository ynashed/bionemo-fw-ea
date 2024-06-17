# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import transformer_engine as te
from megatron.core.transformer.transformer_config import TransformerConfig


class TELayerNorm(te.pytorch.LayerNorm):
    def __init__(self, config: TransformerConfig, *args, **kwargs):
        """A wrapper around transformer engine layernorm that allows it to be initialized with a TransformerConfig.
            This allows this method to be used in a megatron layerspec.

        Args:
            config (TransformerConfig): The megatron config. This is used for extracing sequence_parallel and zero_centered_gamma.
                The rest of the config is not used.
        """
        # Eps tends to get passed through properly, as does hidden_size, but not other params from the config.
        super().__init__(
            *args,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
            sequence_parallel=config.sequence_parallel,
            **kwargs,
        )
