# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Tuple


def num_elements(input_shape: Tuple[int]):
    out = 1
    for i in range(len(input_shape)):
        out *= input_shape[i]
    return out


def shape_excluding(input_shape: Tuple[int], excluded_axes=None):
    number_of_axes = len(input_shape)
    excluded_axes_modded = [i % number_of_axes for i in excluded_axes] if excluded_axes is not None else []
    shape_other_components = [input_shape[i] for i in range(len(input_shape)) if i not in excluded_axes_modded]
    return shape_other_components


def num_elements_excluding(input_shape: Tuple[int], excluded_axes=None):
    shape_other_components = shape_excluding(input_shape, excluded_axes)
    print(shape_other_components)
    return num_elements(shape_other_components)
