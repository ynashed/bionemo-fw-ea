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
import tempfile
from typing import Tuple

import pytest
import pytorch_lightning as plt
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bionemo.data.protein.openfold.datahub import get_structured_paths
from bionemo.data.protein.openfold.datasets import PredictDataset
from bionemo.data.protein.openfold.helpers import collate
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.writer import PredictionFeatureWriter
from bionemo.model.utils import setup_trainer


MSA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../examples/tests/test_data/openfold_data/inference/msas'
)


MSAS = [
    ['7YVT_B_mgnify_alignment.a3m', '7YVT_B_smallbfd_alignment.a3m', '7YVT_B_uniref90_alignment.a3m'],
    ['7ZHL_A_mgnify_alignment.a3m', '7ZHL_A_smallbfd_alignment.a3m', '7ZHL_A_uniref90_alignment.a3m'],
]
MSAS = [[os.path.join(MSA_DIR, msa) for msa in msas] for msas in MSAS]


@pytest.fixture(scope='module')
def infer_cfg() -> DictConfig:
    this_file_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = "examples/protein/openfold/conf"
    config_name = "infer"
    absolute_config_path = os.path.join(os.getenv("BIONEMO_HOME"), config_path)
    relative_config_path = os.path.relpath(absolute_config_path, this_file_dir)
    with initialize(config_path=relative_config_path):
        cfg = compose(config_name=config_name)
    yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture(scope='function')
def alphafold_cfg(request, infer_cfg) -> DictConfig:
    msa_a3m_filepaths, generate_templates_if_missing = request.param
    infer_cfg.model.data.msa_a3m_filepaths = msa_a3m_filepaths
    infer_cfg.model.data.generate_templates_if_missing = generate_templates_if_missing
    return infer_cfg


@pytest.fixture(scope='function')
def alphafold_model_trainer(alphafold_cfg) -> Tuple[AlphaFold, plt.Trainer]:
    trainer = setup_trainer(alphafold_cfg, callbacks=[])
    alphafold = AlphaFold(cfg=alphafold_cfg.model, trainer=trainer)
    return alphafold, trainer


@pytest.mark.parametrize(
    "outputs",
    [
        ['single', 'msa', 'pair', 'sm_single'],
    ],
)
def test_prediction_pdb_writer(infer_cfg, outputs):
    # setup input/output samples for writer
    sequence = infer_cfg.sequences[0]
    seq_name = infer_cfg.seq_names[0]
    N_res = len(sequence)
    bs = infer_cfg.model.micro_batch_size
    N_seq = 508

    input_dict = {'seq_name': [seq_name]}
    output_dict = {
        'single': torch.empty(bs, N_res, infer_cfg.model.evoformer_stack_config.c_s),
        'msa': torch.empty(bs, N_seq, N_res, infer_cfg.model.evoformer_stack_config.c_m),
        'pair': torch.empty(bs, N_res, N_res, infer_cfg.model.evoformer_stack_config.c_z),
        'sm_single': torch.empty(bs, N_res, infer_cfg.model.structure_module_config.c_s),
    }

    # call writer
    with tempfile.TemporaryDirectory() as temp_dir:
        callback = PredictionFeatureWriter(temp_dir, outputs)
        callback.on_predict_batch_end(
            outputs=output_dict,
            batch=input_dict,
            trainer=None,  # dummy
            pl_module=None,  # dummy
            batch_idx=None,  # dummy
        )


def get_predict_dataset(cfg):
    ds_paths = get_structured_paths(cfg.model.data)
    ds = PredictDataset(
        sequences=cfg.sequences,
        seq_names=cfg.seq_names,
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        template_hhr_filepaths=cfg.model.data.template_hhr_filepaths,
        msa_a3m_filepaths=cfg.model.data.msa_a3m_filepaths,
        generate_templates_if_missing=cfg.model.data.generate_templates_if_missing,
        pdb70_database_path=cfg.model.data.pdb70_database_path,
        cfg=cfg.model,
    )
    return ds


@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    "alphafold_cfg",
    [
        (
            [
                [],
            ]
            * len(MSAS),
            False,
        ),  # sequence-only inference
        (MSAS, False),  # sequence-and-msa inference without template
        (MSAS, True),  # inference with template but no template dataset given
    ],
    indirect=True,
)
def test_openfold_inference(alphafold_cfg, alphafold_model_trainer):
    # setup for inference
    if alphafold_cfg.model.data.generate_templates_if_missing:
        with pytest.raises(ValueError):  # raise error if generate template without template database
            ds = get_predict_dataset(alphafold_cfg)
    else:
        alphafold_model, trainer = alphafold_model_trainer
        ds = get_predict_dataset(alphafold_cfg)
        dl = DataLoader(
            ds,
            batch_size=alphafold_cfg.model.micro_batch_size,
            num_workers=alphafold_cfg.model.data.num_workers,
            collate_fn=collate,
        )
        # inference
        trainer.predict(alphafold_model, dl, return_predictions=False)
