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
import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from pytorch_lightning import seed_everything
from rdkit import Chem

from bionemo.data.diffdock.inference import build_inference_datasets
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference, do_inference_sampling
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    check_model_exists,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


e3nn.set_optimization_defaults(optimize_einsums=False)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False

BIONEMO_HOME = os.getenv("BIONEMO_HOME")
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, './conf')
ROOT_DIR = 'diffdock'
CHECKPOINT_PATH = [
    os.path.join(BIONEMO_HOME, "models/molecule/diffdock/diffdock_score.nemo"),
    os.path.join(BIONEMO_HOME, "models/molecule/diffdock/diffdock_confidence.nemo"),
]


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH[0]) and check_model_exists(CHECKPOINT_PATH[1])


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
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH[0])
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH[1])
@pytest.mark.skip(reason="FIXME: Test is flaky across different hardwares")
def test_diffdock_inference(tmp_directory):
    cfg = get_cfg(tmp_directory, PREPEND_CONFIG_DIR, "diffdock_infer_test")
    seed_everything(42, workers=True)

    # process input and build inference datasets for score model, and confidence model, build dataloader
    complex_name_list, test_dataset, confidence_test_dataset, test_loader = build_inference_datasets(cfg)

    # load score model
    model = DiffDockModelInference(cfg.score_infer)
    model.eval()

    # load confidence model
    confidence_model = DiffDockModelInference(cfg.confidence_infer)
    confidence_model.eval()

    do_inference_sampling(
        cfg, model, confidence_model, complex_name_list, test_loader, test_dataset, confidence_test_dataset
    )

    assert os.path.exists(os.path.join(cfg.out_dir, cfg.complex_name))
    results_path = sorted(os.listdir(os.path.join(cfg.out_dir, cfg.complex_name)))[1:]
    assert len(results_path) == cfg.samples_per_complex

    ligand_pos = []
    confidence = []
    for k in range(cfg.samples_per_complex):
        confidence.append(float(results_path[k].split("confidence")[1][:-4]))
        mol_pred = Chem.SDMolSupplier(
            os.path.join(cfg.out_dir, cfg.complex_name, results_path[k]), sanitize=False, removeHs=False
        )[0]
        ligand_pos.append(mol_pred.GetConformer().GetPositions())

    ligand_pos = np.asarray(ligand_pos)
    confidence = np.asarray(confidence)

    ref_ligand = np.load(os.path.join(os.path.dirname(cfg.protein_path), "ref_ligand.npz"))

    assert np.allclose(ligand_pos, ref_ligand['ref_ligand_pos'], atol=1.0e-2)
    assert np.allclose(confidence, ref_ligand['ref_confidence'])
