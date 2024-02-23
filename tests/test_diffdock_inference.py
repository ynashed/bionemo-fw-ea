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

import e3nn
import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from rdkit import Chem

from bionemo.data.diffdock.inference import build_inference_datasets
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference, do_inference_sampling
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import check_model_exists, teardown_apex_megatron_cuda


e3nn.set_optimization_defaults(optimize_einsums=False)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
torch.backends.cudnn.enabled = False


BIONEMO_HOME = os.environ["BIONEMO_HOME"]
CHECKPOINT_PATH = [
    os.path.join(BIONEMO_HOME, "models/molecule/diffdock/diffdock_score.nemo"),
    os.path.join(BIONEMO_HOME, "models/molecule/diffdock/diffdock_confidence.nemo"),
]


@pytest.fixture(scope="function")
def cfg(config_path_for_tests, tmp_path) -> DictConfig:
    cfg = load_model_config(config_name="diffdock_infer_test", config_path=config_path_for_tests)
    cfg.out_dir = tmp_path
    yield cfg
    teardown_apex_megatron_cuda()


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH[0]) and check_model_exists(CHECKPOINT_PATH[1])


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH[0])
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH[1])
def test_diffdock_inference(cfg):
    """
    WARNING: this unit tests is sensitive to the order of the operations. If the models are initiated before
    build_inference_datasets is executed, the test fails
    """
    seed_everything(cfg.seed, workers=True)

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

    assert np.allclose(ligand_pos, ref_ligand['ref_ligand_pos'])
    assert np.allclose(confidence, ref_ligand['ref_confidence'])
