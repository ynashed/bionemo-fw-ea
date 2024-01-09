# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
TEST_DATA_DOWNLOAD_SCRIPT = os.path.join(
    os.environ["BIONEMO_HOME"], 'examples/molecule/diffdock/scripts/download_data_sample.sh'
)
ROOT_DIR = 'diffdock'

inputs = [
    ('diffdock_score_training_test', None, 'USE_FAST_TP'),
    ('diffdock_score_training_test', None, 'marta'),
    ('diffdock_score_training_test', 'SizeAwareBatchSampler', 'USE_FAST_TP'),
    ('diffdock_score_training_test', 'SizeAwareBatchSampler', 'marta'),
    ('diffdock_confidence_training_test', None, 'USE_FAST_TP'),
]


@pytest.fixture(scope="session")
def tmp_directory(tmp_path_factory, root_directory=ROOT_DIR):
    """Create tmp directory"""
    tmp_path_factory.mktemp(root_directory)
    return tmp_path_factory.getbasetemp()


def get_cfg(tmp_directory, prepend_config_path, config_name, config_path='conf'):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    with open_dict(cfg):
        cfg.tmp_directory = tmp_directory

    return cfg


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.parametrize("config_name, batch_sampler, tensor_prodcut_type", inputs)
def test_diffdock_fast_dev_run(tmp_directory, config_name, batch_sampler, tensor_prodcut_type):
    cfg = get_cfg(tmp_directory, PREPEND_CONFIG_DIR, config_name)
    with open_dict(cfg):
        cfg.model.batch_sampler = batch_sampler
    if tensor_prodcut_type == 'USE_FAST_TP':
        os.environ['USE_FAST_TP'] = '1'
    else:
        if 'USE_FAST_TP' in os.environ:
            del os.environ['USE_FAST_TP']
        with open_dict(cfg):
            cfg.model.tensor_product.type = tensor_prodcut_type

    if not os.path.exists(cfg.data.cache_path):
        # download data from ngc
        os.system(f"bash {TEST_DATA_DOWNLOAD_SCRIPT}")

    data_manager = DataManager(cfg)
    trainer = setup_trainer(cfg)
    if "all_atoms" in cfg.data and cfg.data.all_atoms:
        model = AAScoreModel(cfg=cfg, trainer=trainer, data_manager=data_manager)
    else:
        model = CGScoreModel(cfg=cfg, trainer=trainer, data_manager=data_manager)

    trainer.fit(model)
    if 'USE_FAST_TP' in os.environ:
        del os.environ['USE_FAST_TP']
