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
import tempfile

import pytest
import torch
from hydra.experimental import compose, initialize
from torch.utils.data import DataLoader

from bionemo.data.protein.openfold.datahub import get_structured_paths
from bionemo.data.protein.openfold.datasets import PredictDataset
from bionemo.data.protein.openfold.helpers import collate
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.writer import PredictionFeatureWriter
from bionemo.model.utils import setup_trainer
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, '../examples/protein/openfold/conf')
MSA_DIR = os.path.join(THIS_FILE_DIR, '../examples/tests/test_data/openfold_data/inference/msas')

MSAS = [
    ['7ZHL_A_mgnify_alignment.a3m', '7ZHL_A_smallbfd_alignment.a3m', '7ZHL_A_uniref90_alignment.a3m'],
    ['7YVT_B_mgnify_alignment.a3m', '7YVT_B_smallbfd_alignment.a3m', '7YVT_B_uniref90_alignment.a3m'],
]
MSAS = [[os.path.join(MSA_DIR, msa) for msa in msas] for msas in MSAS]


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


@pytest.mark.parametrize(
    "outputs",
    [
        ['single', 'msa', 'pair', 'sm_single'],
    ],
)
def test_prediction_pdb_writer(outputs):
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer')

    # setup input/output samples for writer
    sequence = cfg.sequences[0]
    seq_name = cfg.seq_names[0]
    N_res = len(sequence)
    bs = cfg.model.micro_batch_size
    N_seq = 508

    input_dict = {'seq_name': [seq_name]}
    output_dict = {
        'single': torch.empty(bs, N_res, cfg.model.evoformer_stack_config.c_s),
        'msa': torch.empty(bs, N_seq, N_res, cfg.model.evoformer_stack_config.c_m),
        'pair': torch.empty(bs, N_res, N_res, cfg.model.evoformer_stack_config.c_z),
        'sm_single': torch.empty(bs, N_res, cfg.model.structure_module_config.c_s),
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
    "msa_a3m_filepaths,generate_templates_if_missing",
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
)
def test_openfold_inference(msa_a3m_filepaths, generate_templates_if_missing):
    # setup config
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer')
    cfg.model.data.msa_a3m_filepaths = msa_a3m_filepaths
    cfg.model.data.generate_templates_if_missing = generate_templates_if_missing

    # setup for inference
    if generate_templates_if_missing:
        with pytest.raises(ValueError):  # raise error if generate template without template database
            ds = get_predict_dataset(cfg)
    else:
        trainer = setup_trainer(cfg, callbacks=[])
        alphafold = AlphaFold(cfg=cfg.model, trainer=trainer)  # TODO reduce model size in unittest
        ds = get_predict_dataset(cfg)
        dl = DataLoader(
            ds, batch_size=cfg.model.micro_batch_size, num_workers=cfg.model.data.num_workers, collate_fn=collate
        )

        # inference
        trainer.predict(alphafold, dl, return_predictions=False)
