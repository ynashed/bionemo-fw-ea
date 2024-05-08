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
import subprocess
from glob import glob

import pytest
import torch


BIONEMO_HOME = os.environ["BIONEMO_HOME"]
TEST_DATA_DIR = os.path.join(BIONEMO_HOME, "examples/tests/test_data")


@pytest.fixture
def train_args():
    return {
        'trainer.devices': torch.cuda.device_count(),
        'trainer.num_nodes': 1,
        'trainer.max_steps': 8,
        'trainer.val_check_interval': 2,  # check validation set every 2 training batches
        'trainer.limit_val_batches': 2,  # run validation for 2 validation batches
        'model.data.val.use_upsampling': False,
        'trainer.limit_test_batches': 1,
        'model.data.test.use_upsampling': True,
        'exp_manager.create_wandb_logger': False,
        'exp_manager.create_tensorboard_logger': False,
        'model.micro_batch_size': 2,
        # 'model.data.dataset_path': os.path.join(BIONEMO_HOME, "data")
    }


@pytest.fixture
def data_args():
    return {
        'model.data.dataset.train': 'x000',
        'model.data.dataset.val': 'x000',
        'model.data.dataset.test': 'x000',
    }


# TODO: can we assume that we always run these tests from main bionemo dir?
DIRS_TO_TEST = [
    # 'examples/',
    # 'examples/molecule/megamolbart/',
    # 'examples/protein/downstream/',
    # 'examples/protein/esm1nv/',
    # 'examples/protein/esm2nv/',
    # 'examples/protein/prott5nv/',
    'examples/protein/openfold/',
    # 'examples/dna/dnabert/',
    # 'examples/molecule/diffdock/',
    # 'examples/molecule/molmim/',
]

TRAIN_SCRIPTS = []
for subdir in DIRS_TO_TEST:
    TRAIN_SCRIPTS += list(glob(os.path.join(subdir, '*train*.py')))
    TRAIN_SCRIPTS += [f for f in glob(os.path.join(subdir, 'downstream*.py')) if not f.endswith('test.py')]


def get_data_overrides(script_or_cfg_path: str) -> str:
    """Replace datasets with smaller samples included in the repo

    Based on the script/config file provided, checks what kind of task
    the script performs and selects compatible data sample from test data.
    Returns string that can be appended to the python command for launching the script
    """
    DATA = " ++model.data"
    MAIN = f'{DATA}.dataset_path={TEST_DATA_DIR}/%s'
    DOWNSTREAM = f' ++model.dwnstr_task_validation.dataset.dataset_path={TEST_DATA_DIR}/%s'

    root, domain, model, *conf, script = script_or_cfg_path.split('/')
    assert root == 'examples' and model in (
        'megamolbart',
        'esm1nv',
        'esm2nv',
        'prott5nv',
        'downstream',
        'openfold',
        'dnabert',
        'diffdock',
        'molmim',
    ), 'update this function, patterns might be wrong'

    task = {
        'molecule': 'physchem/SAMPL',
        'protein': 'downstream',
        'dna': 'downstream',
    }

    if conf == ['conf']:
        if model in ('megamolbart', 'openfold', 'molmim'):
            return ''
        else:
            return MAIN % f'{domain}/{task[domain]}/test/x000'

    if 'retro' in script:
        return MAIN % 'reaction'
    elif model == 'openfold':
        return MAIN % 'openfold_data/processed_sample'
    elif model == 'diffdock':
        return (
            f' ++data.split_train={TEST_DATA_DIR}/molecule/diffdock/splits/split_train'
            f' ++data.split_val={TEST_DATA_DIR}/molecule/diffdock/splits/split_val'
            f' ++data.split_test={TEST_DATA_DIR}/molecule/diffdock/splits/split_test'
            f' ++data.cache_path={TEST_DATA_DIR}/molecule/diffdock/data_cache'
        )
    elif 'downstream' in script:
        if model == 'dnabert':
            fasta_directory = os.path.join(TEST_DATA_DIR, 'dna/downstream')
            fasta_pattern = fasta_directory + '/test-chr1.fa'
            splicesite_overrides = (
                f"++model.data.fasta_directory={fasta_directory} "
                "++model.data.fasta_pattern=" + fasta_pattern + " "
                f"++model.data.train_file={fasta_directory}/train.csv "
                f"++model.data.val_file={fasta_directory}/val.csv "
                f"++model.data.predict_file={fasta_directory}/test.csv "
            )
            return splicesite_overrides
        else:
            return MAIN % f'{domain}/{task[domain]}'
    elif model == 'dnabert':
        DNABERT_TEST_DATA_DIR = os.path.join(BIONEMO_HOME, 'examples/dna/dnabert/data/small-example')
        dnabert_overrides = (
            f"++model.data.dataset_path={DNABERT_TEST_DATA_DIR} "
            "++model.data.dataset.train=chr1-trim-train.fna "
            "++model.data.dataset.val=chr1-trim-val.fna "
            "++model.data.dataset.test=chr1-trim-test.fna "
        )
        return dnabert_overrides
    elif model == 'esm2nv' and "infer" not in script:
        # TODO(dorotat) Simplify this case when data-related utils for ESM2 are refactored
        UNIREF_FOLDER = "uniref202104_esm2_qc_test200_val200"
        MAIN = f'{DATA}.train.dataset_path={TEST_DATA_DIR}/%s'
        esm2_overwrites = (
            MAIN % f'{UNIREF_FOLDER}/uf50'
            + f"{DATA}.train.cluster_mapping_tsv={TEST_DATA_DIR}/{UNIREF_FOLDER}/mapping.tsv"
            f"{DATA}.train.index_mapping_dir={TEST_DATA_DIR}/{UNIREF_FOLDER}"
            f"{DATA}.train.uf90.uniref90_path={TEST_DATA_DIR}/{UNIREF_FOLDER}/uf90/"
            f"{DATA}.val.dataset_path={TEST_DATA_DIR}/{UNIREF_FOLDER}/uf50/"
            f"{DATA}.test.dataset_path={TEST_DATA_DIR}/{UNIREF_FOLDER}/uf50/" + DOWNSTREAM % f'{domain}/{task[domain]}'
        )
        return esm2_overwrites

    else:
        return (MAIN + DOWNSTREAM) % (domain, f'{domain}/{task[domain]}')


def get_train_args_overrides(script_or_cfg_path, train_args):
    root, domain, model, *conf, script = script_or_cfg_path.split('/')
    if model == "openfold":
        train_args['model.micro_batch_size'] = 1
        train_args['model.train_ds.num_workers'] = 1
        train_args['model.train_sequence_crop_size'] = 16
        # do not use kalign as it requires third-party-download and it not essential for testing
        train_args['model.data.realign_when_required'] = False
    elif model == "diffdock":
        # Use size aware batch sampler, and set the size control to default
        train_args['model.micro_batch_size'] = 2
        train_args['model.estimate_memory_usage.maximal'] = 'null'
        train_args['model.max_total_size'] = 'null'

    return train_args


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize('script_path', TRAIN_SCRIPTS)
def test_stop_and_go(script_path, train_args, data_args, tmp_path):
    data_str = get_data_overrides(script_path)
    train_args = get_train_args_overrides(script_path, train_args)
    # add kill-after-checkpoint and metadata-save callbacks for first run
    train_args['create_kill_after_signal_callback'] = True
    train_args['kill_after_signal_callback_kwargs.metadata_path'] = tmp_path
    train_args['create_metadata_save_callback'] = True
    train_args['metadata_save_callback_kwargs.metadata_path'] = tmp_path

    cmd = f'python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} ' + ' '.join(
        f'++{k}={v}' for k, v in train_args.items()
    )
    # TODO(dorotat) Trye to simplify when data-related utils for ESM2 are refactored
    if "esm2" not in script_path and "dnabert" not in script_path:
        cmd += ' ' + ' '.join(f'++{k}={v}' for k, v in data_args.items())
    print(cmd)
    # run initial training run to save a checkpoint and some metadata and kill the job afterwards
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    error_out = process_handle.stderr.decode('utf-8')
    assert process_handle.returncode == 0, f"Initial training command failed:\n{cmd}\n Error log:\n{error_out}"

    # assert that metadata was saved correctly
    assert os.path.isfile(tmp_path / 'checkpoints/metadata.pkl'), f"No file found at {tmp_path / 'metadata.pkl'}"

    # add check checkpoint integrity callback for second run
    train_args['create_checkpoint_integrity_callback'] = True
    train_args['checkpoint_integrity_callback_kwargs.metadata_path'] = tmp_path
    # remove kill after checkpoint and metadata save callbacks for second run
    train_args['create_kill_after_signal_callback'] = False
    train_args['kill_after_signal_callback_kwargs'] = None
    train_args['create_metadata_save_callback'] = False
    train_args['metadata_save_callback_kwargs.metadata_path'] = None

    cmd = f'python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} ' + ' '.join(
        f'++{k}={v}' for k, v in train_args.items()
    )
    # TODO(dorotat) Trye to simplify  when data-related utils for ESM2 are refactored
    if "esm2" not in script_path and "dnabert" not in script_path:
        cmd += ' ' + ' '.join(f'++{k}={v}' for k, v in data_args.items())
    print(cmd)
    # run resume training run to load checkpoint and check against saved metadata
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    error_out = process_handle.stderr.decode('utf-8')
    assert process_handle.returncode == 0, f"Resuming training command failed:\n{cmd}\n Error log:\n{error_out}"
