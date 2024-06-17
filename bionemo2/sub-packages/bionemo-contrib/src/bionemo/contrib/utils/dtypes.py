# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Literal

import torch


PrecisionTypes = Literal['fp16', 'bf16', 'fp32', 'bf16-mixed', 'fp32-mixed']


def get_autocast_dtype(precision: PrecisionTypes) -> torch.dtype:
    # TODO move this to a utilities folder, or find/import the function that does this in NeMo
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    elif precision == "fp32":
        return torch.float32
    elif precision == "bf16-mixed":
        return torch.bfloat16
    elif precision == "fp32-mixed":
        return torch.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}")
