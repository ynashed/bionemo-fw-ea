# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import pytest
import torch
from torch.nn import Parameter
from torch.optim import Adam

from bionemo.utils.scheduler import LinearWarmupHoldDecayPolicy


exp_values = [
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.1,
    0.1,
    0.1,
    0.05,
    0.025,
    0,
    0,
]


@pytest.mark.skip(reason="FIXME: this test hangs when executed after test_esm2nv_data_utils.py in pytest")
def test_linear_warmup_hold_decay_scheduler():
    n_steps = 12
    lr = 0.1

    model = [Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = Adam(lr=lr, params=model)

    warmup_steps = 4

    scheduler = LinearWarmupHoldDecayPolicy(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        constant_steps=3,
        max_steps=10,
        min_lr=0,
    )

    lrs = np.zeros(n_steps)
    for i in range(n_steps):
        lrs[i] = scheduler.get_lr()[0]
        scheduler.step()

    np.testing.assert_allclose(lrs, exp_values)
