"""
This file tests the data-related utilities for ESM2.
"""
import os
import pathlib

import pytest
from hydra import compose, initialize

from bionemo.data import mapped_dataset, memmap_csv_fields_dataset
from bionemo.data.preprocess.protein.preprocess import ESM2Preprocess
from bionemo.model.protein.esm1nv import esm1nv_model
from bionemo.model.utils import initialize_distributed_parallel_state, setup_trainer
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


os.environ["BIONEMO_HOME"] = os.environ.get("BIONEMO_HOME", '/workspace/bionemo')


THIS_FILE_DIR = pathlib.Path(os.path.abspath(__file__))
PROJ_BASE_DIR = THIS_FILE_DIR.parent
CONFIG_PATH = "../examples/tests/conf"
PREPEND_CONFIG_DIR = PROJ_BASE_DIR / "examples" / "protein" / "esm2nv" / "conf"


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


@pytest.fixture
def model_and_configs():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='esm2nv_data_test', config_path=CONFIG_PATH)

    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    return model, cfg


@pytest.fixture()
def cfg():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='esm2nv_data_test', config_path=CONFIG_PATH)
    return cfg


def test_esm2nv_model_creates_train_dataset_with_expected_number_of_samples(model_and_configs):
    model, cfg = model_and_configs

    num_train_samples = 10
    train_dataset = model.build_train_dataset(cfg.model, num_train_samples)

    sample = next(iter(train_dataset))

    assert len(train_dataset) == 10
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample['sequence'], str)
    assert isinstance(sample['sequence_id'], str)


def test_esm2nv_model_creates_train_dataset_fails_when_num_samples_is_none(model_and_configs):
    model, cfg = model_and_configs
    num_train_samples = None
    cfg.model.data.train.use_upsampling = True
    with pytest.raises(AssertionError):
        model.build_train_dataset(cfg.model, num_train_samples)


def test_esm2nv_model_creates_validation_dataset_with_valid_outputs_given_num_samples_is_none(model_and_configs):
    model, cfg = model_and_configs

    num_val_samples = None
    val_dataset = model.build_val_dataset(cfg.model, num_val_samples)

    sample = next(iter(val_dataset))

    assert len(val_dataset) == 200
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample['sequence'], str)
    assert isinstance(sample['sequence_id'], str)


def test_esm2nv_model_creates_validation_dataset_with_set_length(cfg):
    num_val_samples = 10
    cfg.model.data.val.use_upsampling = True

    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    val_dataset = model.build_val_dataset(cfg.model, num_val_samples)

    assert len(val_dataset) == num_val_samples
    sample = next(iter(val_dataset))

    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample['sequence'], str)
    assert isinstance(sample['sequence_id'], str)


def test_esm2nv_model_creates_test_dataset_with_valid_outputs(model_and_configs):
    model, cfg = model_and_configs
    cfg.model.data.test.use_upsampling = False

    test_dataset = model.build_test_dataset(cfg.model, None)

    sample = next(iter(test_dataset))
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample['sequence'], str)
    assert isinstance(sample['sequence_id'], str)


def test_esm2nv_model_creates_test_dataset_with_set_length(cfg):
    num_test_samples = 10
    cfg.model.data.test.use_upsampling = True

    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    test_dataset = model.build_test_dataset(cfg.model, num_test_samples)

    assert len(test_dataset) == num_test_samples

    sample = next(iter(test_dataset))
    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample['sequence'], str)
    assert isinstance(sample['sequence_id'], str)


def test_esm2nv_model_build_train_valid_test_datasets_returns_valid_datasets(model_and_configs):
    model, _ = model_and_configs

    train_ds, val_ds, test_ds = model.build_train_valid_test_datasets()

    assert isinstance(train_ds, mapped_dataset.Uniref90ClusterMappingDataset)
    assert isinstance(val_ds, memmap_csv_fields_dataset.CSVFieldsMemmapDataset)
    assert isinstance(test_ds, mapped_dataset.ResamplingMappedDataset)


def test_build_train_valid_test_dataset_limits_train_batches_based_on_max_steps(cfg):
    # num_train_samples = int(max_train_steps * global_batch_size)
    cfg.trainer.max_steps = 5
    cfg.model.micro_batch_size = 4
    initialize_distributed_parallel_state()
    trainer = setup_trainer(cfg, adjust_config=True)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    train_ds, _, _ = model.build_train_valid_test_datasets()

    assert len(train_ds) == 20


def test_esm2nv_model_fails_if_test_use_upsampling_false_but_limit_test_batches_not_1(cfg):
    cfg.trainer.limit_test_batches = 100
    cfg.model.data.test.use_upsampling = False

    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    with pytest.raises(ValueError, match=r"config.model.data.test.use_upsampling is False "):
        model.build_train_valid_test_datasets()


def test_esm2nv_model_fails_if_val_use_upsampling_false_but_limit_val_batches_not_1(cfg):
    cfg.trainer.limit_val_batches = 100
    cfg.model.data.val.use_upsampling = False

    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    with pytest.raises(ValueError, match=r"config.model.data.val.use_upsampling"):
        model.build_train_valid_test_datasets()


def test_build_train_valid_test_dataset_limits_val_batches_uses_fraction_of_dataset(cfg):
    cfg.trainer.limit_val_batches = 0.5
    cfg.trainer.max_steps = 500
    cfg.trainer.val_check_interval = 1
    cfg.model.data.val.use_upsampling = True

    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    _, val_ds, _ = model.build_train_valid_test_datasets()

    val_dataset_length = 200
    expected_len_val_ds = (
        (cfg.trainer.max_steps // cfg.trainer.val_check_interval + 1)
        * ((val_dataset_length * cfg.trainer.limit_val_batches) / model._cfg.global_batch_size)
        * model._cfg.global_batch_size
    )
    assert expected_len_val_ds == 50100.0
    assert len(val_ds) == expected_len_val_ds


def test_build_train_valid_test_dataset_limits_test_batches_uses_fraction_of_dataset(cfg):
    cfg.trainer.limit_test_batches = 0.5
    cfg.model.data.test.use_upsampling = True

    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    _, _, test_ds = model.build_train_valid_test_datasets()

    test_dataset_length = 200
    expected_len_test_ds = test_dataset_length * cfg.trainer.limit_test_batches
    assert expected_len_test_ds == 100
    assert len(test_ds) == expected_len_test_ds


def test_build_train_valid_test_dataset_limits_val_batches_uses_full_dataset(cfg):
    cfg.trainer.limit_val_batches = 1.0
    cfg.trainer.max_steps = 10
    cfg.trainer.val_check_interval = 2
    cfg.model.data.val.use_upsampling = True
    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    _, val_ds, _ = model.build_train_valid_test_datasets()

    val_dataset_length = 200
    expected_len_val_ds = (
        (cfg.trainer.max_steps // cfg.trainer.val_check_interval + 1)
        * ((val_dataset_length * cfg.trainer.limit_val_batches) / model._cfg.global_batch_size)
        * model._cfg.global_batch_size
    )

    assert expected_len_val_ds == 1200.0
    assert len(val_ds) == expected_len_val_ds


def test_build_train_valid_test_dataset_limits_val_batches_int_2(cfg):
    cfg.trainer.limit_val_batches = int(2)
    cfg.trainer.max_steps = 10
    cfg.trainer.val_check_interval = 2
    cfg.model.data.val.use_upsampling = True
    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    _, val_ds, _ = model.build_train_valid_test_datasets()

    expected_len_val_ds = (
        (cfg.trainer.max_steps // cfg.trainer.val_check_interval + 1)
        * model._cfg.global_batch_size
        * cfg.trainer.limit_val_batches
    )

    assert expected_len_val_ds == 24
    assert len(val_ds) == expected_len_val_ds


def test_build_train_valid_test_dataset_limits_val_batches_error_when_zero(cfg):
    cfg.trainer.limit_val_batches = 0
    cfg.model.data.val.use_upsampling = True
    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    with pytest.raises(ValueError, match=r"trainer.limit_val_batches is set to 0"):
        model.build_train_valid_test_datasets()


def test_build_train_valid_test_dataset_throws_error_if_limit_val_batches_but_no_upsampling_in_cfg(cfg):
    cfg.trainer.limit_val_batches = 0.5
    cfg.model.data.val.use_upsampling = False
    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    with pytest.raises(ValueError, match=r"config.model.data.val.use_upsampling"):
        model.build_train_valid_test_datasets()


def test_build_train_valid_test_dataset_limits_test_batches_uses_batch_num_specified_no_scaling(cfg):
    # Note: The test dataset does not really work at all in any capacity.
    cfg.trainer.max_steps = 10
    cfg.trainer.limit_test_batches = 200
    cfg.model.data.test.use_upsampling = True
    initialize_distributed_parallel_state()

    trainer = setup_trainer(cfg)
    trainer.limit_test_batches = 200
    model = esm1nv_model.ESM2nvModel(cfg.model, trainer)
    _, _, test_ds = model.build_train_valid_test_datasets()

    assert len(test_ds) == model._cfg.global_batch_size * trainer.limit_test_batches
    assert len(test_ds) == 400


# TODO: Move to Dataprep testing once its available.
def test_esm2nv_preprocess_training_dataset_creates_non_empty_dirs():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='esm2nv_data_test', config_path=CONFIG_PATH)
    preprocessor = ESM2Preprocess()

    preprocessor.prepare_dataset(
        uf50_datapath=cfg.model.data.train.uf50_datapath,
        uf90_datapath=cfg.model.data.train.uf90_datapath,
        cluster_mapping_tsv=cfg.model.data.train.cluster_mapping_tsv,
        uf50_output_dir="/tmp/uf50_output_dir_train",
        uf90_output_dir="/tmp/uf90_output_dir_train",
        sort_fastas=cfg.model.data.train.sort_fastas,
        mode="train",
    )

    # Make sure the resulting directory is non empty.
    assert os.listdir("/tmp/uf50_output_dir_train")
    assert os.listdir("/tmp/uf90_output_dir_train")


def test_esm2nv_preprocess_val_dataset_creates_non_empty_dirs():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='esm2nv_data_test', config_path=CONFIG_PATH)
    preprocessor = ESM2Preprocess()
    preprocessor.prepare_dataset(
        uf50_datapath=cfg.model.data.val.uf50_datapath,
        uf50_output_dir="/tmp/uf50_output_dir_val",
        sort_fastas=False,
        mode="val",
    )
    assert os.listdir("/tmp/uf50_output_dir_val")


def test_esm2nv_preprocess_test_dataset_creates_non_empty_dirs():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='esm2nv_data_test', config_path=CONFIG_PATH)
    preprocessor = ESM2Preprocess()
    preprocessor.prepare_dataset(
        uf50_datapath=cfg.model.data.test.uf50_datapath,
        uf50_output_dir="/tmp/uf50_output_dir_test",
        sort_fastas=False,
        mode="test",
    )
    assert os.listdir("/tmp/uf50_output_dir_test")
