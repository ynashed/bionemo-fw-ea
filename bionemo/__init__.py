# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from pathlib import Path
from typing import Sequence

from nemo.core.optim.lr_scheduler import register_scheduler

from bionemo.utils.scheduler import (
    LinearWarmupHoldDecayParams,
    LinearWarmupHoldDecayPolicy,
)


__all__: Sequence[str] = ("BIONEMO_PY_ROOT",)

# TODO [mgreaves]: Refactor, don't modify global state on bionemo import.
#                  Instead, perform this registration on a per-model-train basis.
register_scheduler(
    'LinearWarmupHoldDecayPolicy',
    LinearWarmupHoldDecayPolicy,
    LinearWarmupHoldDecayParams,
)

BIONEMO_PY_ROOT: Path = Path(__file__).parent
"""The directory containg the bionemo Python package.
"""
