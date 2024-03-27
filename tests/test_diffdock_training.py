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
from typing import Union

import e3nn
import pytest
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from bionemo.data.diffdock.data_manager import DataManager
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModel as CGScoreModel,
)
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModelAllAtom as AAScoreModel,
)
from bionemo.model.utils import setup_trainer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
e3nn.set_optimization_defaults(optimize_einsums=True)

inputs = [
    ('diffdock_score_training_test', None, 'fast_tp'),
    ('diffdock_score_training_test', 'SizeAwareBatchSampler', 'fast_tp'),
    ('diffdock_score_training_test', 'SizeAwareBatchSampler', 'marta'),
    ('diffdock_confidence_training_test', None, 'fast_tp'),
]


@pytest.fixture(scope="function", params=inputs)
def diffdock_cfg(request, tmp_path, config_path_for_tests) -> DictConfig:
    config_name, batch_sampler, tensor_product_type = request.param
    cfg = load_model_config(config_name=config_name, config_path=config_path_for_tests)
    with open_dict(cfg):
        cfg.exp_manager.exp_dir = tmp_path
        cfg.model.batch_sampler = batch_sampler
        cfg.model.tensor_product.type = tensor_product_type
    return cfg


@pytest.fixture(scope="function")
def trainer(diffdock_cfg) -> Trainer:
    trainer = setup_trainer(diffdock_cfg)
    return trainer


@pytest.fixture(scope="function")
def data_manager(diffdock_cfg) -> DataManager:
    data_manager = DataManager(diffdock_cfg)
    yield data_manager
    DataManager.reset_instances()


@pytest.fixture(scope="function")
def model(diffdock_cfg, trainer, data_manager) -> Union[AAScoreModel, CGScoreModel]:
    if "all_atoms" in diffdock_cfg.data and diffdock_cfg.data.all_atoms:
        model = AAScoreModel(cfg=diffdock_cfg, trainer=trainer, data_manager=data_manager)
    else:
        model = CGScoreModel(cfg=diffdock_cfg, trainer=trainer, data_manager=data_manager)
    yield model
    teardown_apex_megatron_cuda()


@pytest.mark.slow
@pytest.mark.needs_gpu
def test_diffdock_fast_dev_run(model: Union[AAScoreModel, CGScoreModel], trainer: Trainer):
    trainer.fit(model)
