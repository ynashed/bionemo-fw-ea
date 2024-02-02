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
import pathlib

import e3nn
import pytest
from hydra import compose, initialize
from omegaconf import open_dict

from bionemo.data.diffdock.data_manager import DataManager
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModel as CGScoreModel,
)
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModelAllAtom as AAScoreModel,
)
from bionemo.model.utils import setup_trainer
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
e3nn.set_optimization_defaults(optimize_einsums=True)

THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, './conf')
ROOT_DIR = 'diffdock'

inputs = [
    ('diffdock_score_training_test', None, 'fast_tp'),
    ('diffdock_score_training_test', 'SizeAwareBatchSampler', 'fast_tp'),
    ('diffdock_score_training_test', 'SizeAwareBatchSampler', 'marta'),
    ('diffdock_confidence_training_test', None, 'fast_tp'),
]


def get_cfg(tmp_path, prepend_config_path, config_name, config_path='conf'):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    with open_dict(cfg):
        cfg.tmp_path = tmp_path

    return cfg


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.parametrize("config_name, batch_sampler, tensor_prodcut_type", inputs)
def test_diffdock_fast_dev_run(tmp_path, config_name, batch_sampler, tensor_prodcut_type):
    cfg = get_cfg(tmp_path, PREPEND_CONFIG_DIR, config_name)
    with open_dict(cfg):
        cfg.model.batch_sampler = batch_sampler
        cfg.model.tensor_product.type = tensor_prodcut_type

    DataManager.reset_instances()
    data_manager = DataManager(cfg)
    trainer = setup_trainer(cfg)
    if "all_atoms" in cfg.data and cfg.data.all_atoms:
        model = AAScoreModel(cfg=cfg, trainer=trainer, data_manager=data_manager)
    else:
        model = CGScoreModel(cfg=cfg, trainer=trainer, data_manager=data_manager)

    trainer.fit(model)
    DataManager.reset_instances()
