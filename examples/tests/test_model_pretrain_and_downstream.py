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
In order to update test results or configs please use the following command:

UPDATE_EXPECTED_RESULTS=1 pytest examples/tests/test_model_pretrain_and_downstream.py
UPDATE_EXPECTED_CFG=1 pytest examples/tests/test_model_pretrain_and_downstream.py
"""

import logging
import os
import pathlib

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from bionemo.callbacks import setup_dwnstr_task_validation_callbacks
from bionemo.data.diffdock.data_manager import DataManager as DiffdockDataManagers
from bionemo.data.equidock import DataManager
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModel as DiffdockScoreModel,
)
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModelAllAtom as DiffdockConfidenceModel,
)
from bionemo.model.molecule.diffdock.setup_trainer import DiffdockTrainerBuilder
from bionemo.model.molecule.megamolbart import FineTuneMegaMolBART, MegaMolBARTModel, MegaMolBARTRetroModel
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.protein.equidock.equidock_model import EquiDock
from bionemo.model.protein.esm1nv import ESM1nvModel, ESM2nvModel
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.prott5nv import ProtT5nvModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    check_expected_training_results,
    check_model_exists,
    clean_directory,
    load_expected_training_results,
    register_searchpath_config_plugin,
    resolve_cfg,
    save_expected_training_results,
    update_relative_config_dir,
)


# logger
logging.getLogger('nemo_logger').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Pretraining, encoder finetuning and secondary structure validation-in-the-loop tests

PREPEND_CONFIG_DIR = [
    '../molecule/megamolbart/conf',
    '../molecule/megamolbart/conf',
    '../protein/esm1nv/conf',
    '../protein/prott5nv/conf',
    '../protein/prott5nv/conf',
    '../protein/esm1nv/conf',
    '../protein/prott5nv/conf',
    '../molecule/megamolbart/conf',
    '../molecule/diffdock/conf',
    '../molecule/diffdock/conf',
    '../protein/equidock/conf',
    '../protein/equidock/conf',
    '../protein/openfold/conf',
    '../protein/esm2nv/conf',
]
CONFIG_NAME = [
    'megamolbart_downstream_retro_test',
    'megamolbart_test',
    'esm1nv_test',
    'prott5nv_test',
    'prott5nv_encoder_finetune_test',
    'esm1nv_encoder_finetune_test',
    'prott5nv_sec_str_val_test',
    'megamolbart_physchem_test',
    'diffdock_score_test',
    'diffdock_confidence_test',
    'equidock_pretrain_test',
    'equidock_finetune_test',
    'openfold_initial_training_test',
    'esm2nv_8M_test',
]
CORRECT_CONFIG = [
    'megamolbart_retro_config',
    'megamolbart_config',
    'esm1nv_config',
    'prott5nv_config',
    'prott5nv_encoder_finetune_config',
    'esm1nv_encoder_finetune_config',
    'prott5nv_sec_str_val_config',
    'megamolbart_physchem_config',
    'diffdock_score_config',
    'diffdock_confidence_config',
    'equidock_pretrain_config',
    'equidock_finetune_config',
    'openfold_initial_training_config',
    'esm2nv_8M_config',
]
CORRECT_RESULTS = [
    'megamolbart_retro_log.json',
    'megamolbart_log.json',
    'esm1nv_log.json',
    'prott5nv_log.json',
    'prott5nv_encoder_finetune_log.json',
    'esm1nv_encoder_finetune_log.json',
    'prott5nv_sec_str_val_log.json',
    'megamolbart_physchem_log.json',
    'diffdock_score_log.json',
    'diffdock_confidence_log.json',
    'equidock_pretrain_log.json',
    'equidock_finetune_log.json',
    'openfold_initial_training_log.json',
    'esm2nv_8M_log.json',
]
MODEL_CLASS = [
    MegaMolBARTRetroModel,
    MegaMolBARTModel,
    ESM1nvModel,
    ProtT5nvModel,
    FineTuneProteinModel,
    FineTuneProteinModel,
    ProtT5nvModel,
    FineTuneMegaMolBART,
    DiffdockScoreModel,
    DiffdockConfidenceModel,
    EquiDock,
    EquiDock,
    AlphaFold,
    ESM2nvModel,
]
MODEL_PARAMETERS = [
    45058048,
    4146176,
    43612544,
    198970496,
    199145485,
    43787533,
    198970496,
    66817,
    20248214,
    4769636,
    525671,
    684074,
    93229082,
    7542848,
]


THIS_FILE_DIR = pathlib.Path(os.path.abspath(__file__)).parent


def get_cfg(prepend_config_path, config_name, config_path='conf'):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)

    if GlobalHydra.instance().is_initialized():
        # in case another hydra is running
        GlobalHydra.instance().clear()

    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    return cfg


@pytest.mark.parametrize('prepend_config_path', PREPEND_CONFIG_DIR)
def test_relative_param_config_path(prepend_config_path):
    '''Ensure the relative config path required by hydra is resolving correctly'''

    updated_prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)
    assert os.path.exists(updated_prepend_config_dir)


@pytest.mark.parametrize(
    'prepend_config_path, config_name, correct_config', list(zip(PREPEND_CONFIG_DIR, CONFIG_NAME, CORRECT_CONFIG))
)
def test_config_parameters(prepend_config_path, config_name, correct_config):
    '''Load the config parameters and ensure they're identical to previous'''

    cfg = get_cfg(prepend_config_path, config_name)
    results_comparison_dir = os.path.abspath(os.path.join(THIS_FILE_DIR, 'expected_results'))

    if os.environ.get('UPDATE_EXPECTED_CFG', False):
        msg = f'Updating expected config in {results_comparison_dir}/{correct_config}'
        logger.warning(msg)
        # will create a new comparison config
        OmegaConf.save(cfg, os.path.join(results_comparison_dir, f"{correct_config}.yaml"))
        assert False, msg

    original_cfg_dict = resolve_cfg(get_cfg(results_comparison_dir, correct_config))
    new_cfg_dict = resolve_cfg(cfg)
    assert (
        original_cfg_dict == new_cfg_dict
    ), f"Mismatch in config {results_comparison_dir}/{correct_config}.\nIn order to update please use the folllowing command:\n UPDATE_EXPECTED_CFG=1 pytest examples/tests/test_model_pretrain_and_downstream.py"


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.parametrize(
    'prepend_config_path, config_name, model_class, model_parameters',
    list(zip(PREPEND_CONFIG_DIR, CONFIG_NAME, MODEL_CLASS, MODEL_PARAMETERS)),
)
def test_model_size(prepend_config_path, config_name, model_class, model_parameters):
    '''Check that number of model weights are correct'''

    cfg = get_cfg(prepend_config_path, config_name)
    callbacks = setup_dwnstr_task_validation_callbacks(cfg)

    trainer = setup_trainer(cfg, callbacks=callbacks)
    if model_class == FineTuneProteinModel or model_class == FineTuneMegaMolBART:
        model = model_class(cfg, trainer)
    elif model_class == DiffdockScoreModel or model_class == DiffdockConfidenceModel:
        data_manager = DiffdockDataManagers(cfg)
        trainer = setup_trainer(cfg, builder=DiffdockTrainerBuilder)
        model = model_class(cfg=cfg, trainer=trainer, data_manager=data_manager)

    elif model_class == EquiDock:
        data_manager = DataManager(cfg)
        cfg.model.input_edge_feats_dim = data_manager.train_ds[0][0].edata['he'].shape[1]
        trainer = setup_trainer(cfg, callbacks=callbacks)
        model = EquiDock(cfg=cfg, trainer=trainer, data_manager=data_manager)

    else:
        model = model_class(cfg.model, trainer)
    assert model.num_weights == model_parameters


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    'prepend_config_path, config_name, model_class, correct_results',
    list(zip(PREPEND_CONFIG_DIR, CONFIG_NAME, MODEL_CLASS, CORRECT_RESULTS)),
)
def test_model_training(prepend_config_path, config_name, model_class, correct_results):
    '''Run short model training and ensure key metrics are identical'''
    seed_everything(0, workers=True)
    cfg = get_cfg(prepend_config_path, config_name)
    clean_directory(cfg.exp_manager.exp_dir)
    callbacks = setup_dwnstr_task_validation_callbacks(cfg)
    trainer = setup_trainer(cfg, callbacks=callbacks)

    if model_class == MegaMolBARTRetroModel:
        pretrain_model_path = cfg.get('restore_from_path', None)
        check_model_exists(pretrain_model_path)
        model = model_class.restore_from(
            restore_path=pretrain_model_path,
            trainer=trainer,
            save_restore_connector=BioNeMoSaveRestoreConnector(),
            override_config_path=cfg,
        )
    elif model_class == DiffdockScoreModel or model_class == DiffdockConfidenceModel:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        data_manager = DiffdockDataManagers(cfg)
        trainer = setup_trainer(cfg, builder=DiffdockTrainerBuilder)
        model = model_class(cfg=cfg, trainer=trainer, data_manager=data_manager)
    elif model_class == EquiDock:
        data_manager = DataManager(cfg)
        cfg.model.input_edge_feats_dim = data_manager.train_ds[0][0].edata['he'].shape[1]
        trainer = setup_trainer(cfg, callbacks=callbacks)
        model = EquiDock(cfg=cfg, trainer=trainer, data_manager=data_manager)
    else:
        if model_class == FineTuneProteinModel or model_class == FineTuneMegaMolBART:
            model = model_class(cfg, trainer)
        else:
            model = model_class(cfg.model, trainer)
    trainer.fit(model)

    results_comparison_dir = os.path.abspath(os.path.join(THIS_FILE_DIR, 'expected_results'))
    trainer_results = trainer.logged_metrics
    if os.environ.get('UPDATE_EXPECTED_RESULTS', False):
        # update only the keys that are in the current results
        msg = f'Updating expected results in {results_comparison_dir}/{correct_results}'
        logger.warning(msg)

        try:
            expected_results_keys = load_expected_training_results(results_comparison_dir, correct_results).keys()

        except FileNotFoundError:
            # if expected results are not found and UPDATE_EXPECTED_RESULTS is True,
            # the script generates expected results
            logger.warning('Expected results not found. Saving all training metrics')
            expected_results_keys = trainer_results.keys()

        save_expected_training_results(
            results_comparison_dir, correct_results, {k: trainer_results[k].item() for k in expected_results_keys}
        )
        assert False, msg

    expected_results = load_expected_training_results(results_comparison_dir, correct_results)

    check_expected_training_results(
        trainer_results,
        expected_results,
        err_msg="\nIn order to update please use the folllowing command:\n UPDATE_EXPECTED_RESULTS=1 pytest examples/tests/test_model_pretrain_and_downstream.py",
    )
    assert True
