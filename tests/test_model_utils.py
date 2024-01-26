# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from io import StringIO

import pytest
from omegaconf.omegaconf import OmegaConf

from bionemo.model.utils import TrainerBuilder


cfg_template = """
do_training: {do_training}
trainer:
    devices: {devices}
    num_nodes: {num_nodes}
    accumulate_grad_batches: {accumulate_grad_batches}
model:
    global_batch_size: {global_batch_size}
    micro_batch_size: {micro_batch_size}
    tensor_model_parallel_size: {tensor_model_parallel_size}
    pipeline_model_parallel_size: {pipeline_model_parallel_size}
"""


def load_cfg_str(cfg):
    return OmegaConf.load(StringIO(cfg))


def test_adjust_config_no_keys():
    cfg = load_cfg_str(
        cfg_template.format(
            do_training=False,
            devices=6,
            num_nodes=8,
            accumulate_grad_batches=5,
            global_batch_size=240,
            micro_batch_size=7,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=3,
        )
    )

    TrainerBuilder.adjust_config(cfg)
    assert cfg.model.global_batch_size == 280


def test_adjust_config_errors_on_disagreement():
    cfg = load_cfg_str(
        cfg_template.format(
            do_training=True,
            devices=6,
            num_nodes=8,
            accumulate_grad_batches=5,
            global_batch_size=240,
            micro_batch_size=7,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=3,
        )
    )

    with pytest.raises(ValueError):
        TrainerBuilder.adjust_config(cfg)
