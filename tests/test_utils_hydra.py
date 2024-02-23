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
import random
import string

import pytest
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from bionemo.utils.hydra import load_model_config


def generate_random_key(num_keys: int):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(num_keys))


def generate_random_value(num_keys: int):
    return random.choice(
        [random.randint(1, 100), random.uniform(0.0, 1.0), ''.join(random.choices(string.ascii_letters, k=num_keys))]
    )


def generate_random_omegaconf_dict(depth: int, num_keys: int):
    if depth == 0:
        return generate_random_value(num_keys=num_keys)
    return OmegaConf.create(
        {
            generate_random_key(num_keys=num_keys): generate_random_omegaconf_dict(depth=depth - 1, num_keys=num_keys)
            for _ in range(num_keys)
        }
    )


@pytest.fixture(scope="module")
def config():
    seed_everything(42)
    return generate_random_omegaconf_dict(depth=2, num_keys=3)


def test_load_model_config(tmp_path, config):
    config_name = "config"
    OmegaConf.save(config, os.path.join(tmp_path, config_name + ".yaml"))

    config_loaded = load_model_config(config_name=config_name, config_path=str(tmp_path))
    assert config_loaded == config
