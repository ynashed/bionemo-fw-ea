# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import logging

import numpy as np
import numpy.testing as npt
import torch
from nemo.core.optim.lr_scheduler import get_scheduler, register_scheduler
from torch.nn import Parameter
from torch.optim import Adam

from bionemo.utils.scheduler import LinearWarmupHoldDecayParams, LinearWarmupHoldDecayPolicy


# needed to register the scheduler


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


def test_linear_warmup_hold_decay_scheduler():
    try:
        register_scheduler(
            name='LinearWarmupHoldDecayPolicy',
            scheduler=LinearWarmupHoldDecayPolicy,
            scheduler_params=LinearWarmupHoldDecayParams,
        )
    except ValueError as e:
        if "Cannot override pre-existing schedulers" in str(e):
            logging.warning("The scheduler has been already registered")
        else:
            raise e

    scheduler_cls = get_scheduler('LinearWarmupHoldDecayPolicy')
    n_steps = 12
    lr = 0.1

    model = [Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = Adam(lr=lr, params=model)

    warmup_steps = 4

    scheduler = scheduler_cls(
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

    npt.assert_allclose(lrs, exp_values)
