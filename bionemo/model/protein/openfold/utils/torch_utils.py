# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from typing import Callable

import torch

from bionemo.data.protein.openfold.helpers import map_tree_leaves


TORCH_SEED_MODULUS = 0xFFFF_FFFF_FFFF_FFFF + 1


def enable_tf32() -> None:
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def disable_tf32() -> None:
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def is_autocast_fp16_enabled() -> bool:
    return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


def map_tensor_tree(fn: Callable, tree: dict) -> dict:
    """Maps tensor tree using given function."""
    return map_tree_leaves(fn=fn, tree=tree, leaf_type=torch.Tensor)
