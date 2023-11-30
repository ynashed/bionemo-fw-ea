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


os.environ["PROJECT_MOUNT"] = os.environ.get("PROJECT_MOUNT", '/workspace/bionemo')

PREPEND_DIR = "../examples/tests/"
CONFIG_PATH = "../examples/protein/esm1nv/conf"
PREPEND_CONFIG_DIR = os.path.abspath("../examples/conf")

THIS_FILE_DIR = pathlib.Path(os.path.abspath(__file__)).parent
PREPEND_DIR = "../examples/tests/"
CONFIG_PATH = os.path.join(PREPEND_DIR, 'conf')
PREPEND_CONFIG_DIR = '../examples/protein/esm1nv/conf'


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


def test_esm2nv_model_creates_train_dataset_with_valid_outputs(model_and_configs):
    model, cfg = model_and_configs

    num_train_samples = 10
    train_dataset = model.build_train_dataset(cfg.model, num_train_samples)

    sample = next(iter(train_dataset))

    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample['sequence'], str)
    assert isinstance(sample['sequence_id'], str)


def test_esm2nv_model_creates_train_dataset_fails_when_num_samples_is_none(model_and_configs):
    model, cfg = model_and_configs

    num_train_samples = None
    with pytest.raises(AssertionError):
        model.build_train_dataset(cfg.model, num_train_samples)


def test_esm2nv_model_creates_validation_dataset_with_valid_outputs_given_num_samples_is_none(model_and_configs):
    model, cfg = model_and_configs

    num_val_samples = None
    val_dataset = model.build_val_dataset(cfg.model, num_val_samples)

    sample = next(iter(val_dataset))

    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample['sequence'], str)
    assert isinstance(sample['sequence_id'], str)


def test_esm2nv_model_creates_validation_dataset_with_valid_outputs(model_and_configs):
    model, cfg = model_and_configs

    num_val_samples = 10
    val_dataset = model.build_val_dataset(cfg.model, num_val_samples)

    sample = next(iter(val_dataset))

    assert sample is not None
    assert isinstance(sample, dict)
    assert isinstance(sample['sequence'], str)
    assert isinstance(sample['sequence_id'], str)


def test_esm2nv_model_creates_test_dataset_with_valid_outputs(model_and_configs):
    model, cfg = model_and_configs

    num_test_samples = 10
    test_dataset = model.build_test_dataset(cfg.model, num_test_samples)

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
    assert isinstance(test_ds, memmap_csv_fields_dataset.CSVFieldsMemmapDataset)


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
