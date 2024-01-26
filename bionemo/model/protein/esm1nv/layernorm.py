# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm
from nemo.utils import logging


def esm_get_layer_norm(normalized_shape, *args, **kwargs):
    # TODO(srabhi, georgea): refactor the custom esm_get_layer_norm module using Megatron Core when NeMo 1.21 is available
    use_pt_layernorm = kwargs.pop('use_pt_layernorm', False)
    if use_pt_layernorm:
        logging.warning("Using PyTorch LayerNorm instead of the default NeMo version")
        eps = kwargs.pop('eps', 1e-05)
        return torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True)
    else:
        return get_layer_norm(normalized_shape, *args, **kwargs)
