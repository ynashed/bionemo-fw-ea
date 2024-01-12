# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Callable

import numpy as np

from bionemo.data.protein.openfold.helpers import map_tree_leaves


NUMPY_SEED_MODULUS = 0xFFFF_FFFF + 1


def map_array_tree(fn: Callable, tree: dict) -> dict:
    """Maps array tree using given function."""
    return map_tree_leaves(fn=fn, tree=tree, leaf_type=np.ndarray)
