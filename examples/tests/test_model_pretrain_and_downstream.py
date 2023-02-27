# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
In order to update test results please use the following command:

UPDATE_EXPECTED_RESULTS=1 pytest examples/tests/test_model_pretrain_and_downstream.py
"""
import pytest
import os
import pathlib
from hydra import initialize, compose
import logging

from bionemo.model.molecule.megamolbart import MegaMolBARTModel
from bionemo.model.protein.esm1nv import ESM1nvModel
from bionemo.model.protein.prott5nv import ProtT5nvModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.callbacks.callback_utils import setup_callbacks
from bionemo.utils.tests import ( BioNemoSearchPathConfig,
                                  register_searchpath_config_plugin,
                                  update_relative_config_dir,
                                  resolve_cfg, 
                                  pickle_cfg,
                                  load_cfg_pickle,
                                  clean_directory,
                                  load_expected_training_results,
                                  save_expected_training_results,
                                  check_expected_training_results )

# logger
logger = logging.getLogger(__name__)

# Pretraining and secondary structure prediction tests
PREPEND_CONFIG_DIR = ['../molecule/megamolbart/conf', '../protein/esm1nv/conf', '../protein/prott5nv/conf', '../protein/prott5nv/conf']
CONFIG_NAME = ['megamolbart_test', 'esm1nv_test', 'prott5nv_test', 'prott5nv_sspred_test']
CORRECT_CONFIG = ['megamolbart_config.pkl', 'esm1nv_config.pkl', 'prott5nv_config.pkl', 'prott5nv_sspred_config.pkl']
CORRECT_RESULTS = ['megamolbart_log.json', 'esm1nv_log.json', 'prott5nv_log.json', 'prott5nv_sspred_log.json']
MODEL_CLASS = [MegaMolBARTModel, ESM1nvModel, ProtT5nvModel, ProtT5nvModel]
MODEL_PARAMETERS = [4146176, 43612544, 198970496, 198970496]

####

os.environ['PROJECT_MOUNT'] = os.environ.get('PROJECT_MOUNT', '/workspace/bionemo')
THIS_FILE_DIR = pathlib.Path(os.path.abspath(__file__)).parent

def get_cfg(prepend_config_path, config_name, config_path='conf'):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    return cfg


@pytest.mark.parametrize('prepend_config_path', PREPEND_CONFIG_DIR)
def test_relative_param_config_path(prepend_config_path):
    '''Ensure the relative config path required by hydra is resolving correctly'''

    updated_prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)
    assert os.path.exists(updated_prepend_config_dir)


@pytest.mark.parametrize('prepend_config_path, config_name, correct_config', 
                         list(zip(PREPEND_CONFIG_DIR, CONFIG_NAME, CORRECT_CONFIG)))
def test_config_parameters(prepend_config_path, config_name, correct_config):
    '''Load the config parameters and ensure they're identical to previous'''
    
    cfg = get_cfg(prepend_config_path, config_name)

    results_comparison_dir = os.path.abspath(os.path.join(THIS_FILE_DIR, 'expected_results'))
    # pickle_cfg(cfg, results_comparison_dir, correct_config) # will create a new comparison config

    original_cfg_dict = load_cfg_pickle(results_comparison_dir, correct_config)
    new_cfg_dict = resolve_cfg(cfg)
    assert original_cfg_dict == new_cfg_dict


@pytest.mark.needs_gpu
@pytest.mark.parametrize('prepend_config_path, config_name, model_class, model_parameters', 
                         list(zip(PREPEND_CONFIG_DIR, CONFIG_NAME, MODEL_CLASS, MODEL_PARAMETERS)))
def test_model_size(prepend_config_path, config_name, model_class, model_parameters):
    '''Check that number of model weights are correct'''

    cfg = get_cfg(prepend_config_path, config_name)
    callbacks = setup_callbacks(cfg)

    trainer = setup_trainer(cfg, callbacks=callbacks)
    model = model_class(cfg.model, trainer)
    assert model.num_weights == model_parameters


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.parametrize('prepend_config_path, config_name, model_class, correct_results', 
                         list(zip(PREPEND_CONFIG_DIR, CONFIG_NAME, MODEL_CLASS, CORRECT_RESULTS)))
def test_model_training(prepend_config_path, config_name, model_class, correct_results):
    '''Run short model training and ensure key metrics are identical'''

    cfg = get_cfg(prepend_config_path, config_name)
    clean_directory(cfg.exp_manager.exp_dir)

    callbacks = setup_callbacks(cfg)
    trainer = setup_trainer(cfg, callbacks=callbacks)

    model = model_class(cfg.model, trainer)
    trainer.fit(model)

    results_comparison_dir = os.path.abspath(os.path.join(THIS_FILE_DIR, 'expected_results'))
    expected_results = load_expected_training_results(results_comparison_dir, correct_results)
    if os.environ.get('UPDATE_EXPECTED_RESULTS', False):
        logger.warning(f'Updating expected results in {results_comparison_dir}/{correct_results}')
        save_expected_training_results(results_comparison_dir, correct_results, expected_results)
    trainer_results = trainer.logged_metrics
    check_expected_training_results(
        trainer_results, 
        expected_results,
        err_msg="\nIn order to update please use the folllowing command:\n UPDATE_EXPECTED_RESULTS=1 pytest examples/tests/test_model_pretrain_and_downstream.py"
    )
    assert True
