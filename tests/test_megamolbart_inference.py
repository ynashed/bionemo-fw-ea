# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import pathlib
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from rdkit import Chem

from bionemo.model.molecule.megamolbart import MegaMolBARTInference
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    check_model_exists,
    register_searchpath_config_plugin,
    reset_microbatch_calculator,
    update_relative_config_dir,
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

CHECKPOINT_PATH = os.path.join(os.getenv("BIONEMO_HOME"), "models/molecule/megamolbart/megamolbart.nemo")

_SMIS = [
    'c1cc2ccccc2cc1',
    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
]


@pytest.fixture(scope='module')
def infer_cfg() -> DictConfig:
    config_path = "examples/molecule/megamolbart/conf"
    config_name = "infer"
    prepend_config_dir = os.path.join(os.getenv("BIONEMO_HOME"), "examples/conf")
    this_file_dir = pathlib.Path(pathlib.Path(os.path.abspath(__file__)).parent)
    absolute_config_path = os.path.join(os.getenv("BIONEMO_HOME"), config_path)
    relative_config_path = os.path.relpath(absolute_config_path, this_file_dir)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(Path(prepend_config_dir), this_file_dir)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=relative_config_path):
        cfg = compose(config_name=config_name)
    yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture(scope='module')
def inferer(infer_cfg: DictConfig) -> MegaMolBARTInference:
    # TODO to remove the first reset in the future - test imp should ensire teardown after model is used
    reset_microbatch_calculator()
    yield MegaMolBARTInference(infer_cfg)
    reset_microbatch_calculator()


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH)


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_smis_to_hiddens(inferer: MegaMolBARTInference):
    hidden_state, pad_masks = inferer.seq_to_hiddens(_SMIS)
    assert hidden_state is not None
    assert hidden_state.shape[0] == len(_SMIS)
    assert hidden_state.shape[2] == inferer.model.cfg.max_position_embeddings
    assert pad_masks is not None


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_smis_to_embedding(inferer: MegaMolBARTInference):
    embedding = inferer.seq_to_embeddings(_SMIS)
    assert embedding is not None
    assert embedding.shape[0] == len(_SMIS)
    assert embedding.shape[1] == inferer.model.cfg.max_position_embeddings


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_hidden_to_smis(inferer: MegaMolBARTInference):
    hidden_state, pad_masks = inferer.seq_to_hiddens(_SMIS)
    infered_smis = inferer.hiddens_to_seq(hidden_state, pad_masks)
    log.info(f'Input SMILES and Infered: {_SMIS}, {infered_smis}')

    assert len(infered_smis) == len(_SMIS)

    for smi, infered_smi in zip(_SMIS, infered_smis):
        log.info(f'Input and Infered:{smi},  {infered_smi}')
        input_mol = Chem.MolFromSmiles(smi)
        infer_mol = Chem.MolFromSmiles(infered_smi)
        assert input_mol is not None and infer_mol is not None

        canonical_smi = Chem.MolToSmiles(input_mol, canonical=True)
        canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
        log.info(f'Canonical Input and Infered: {canonical_smi}, {canonical_infered_smi}')

        assert canonical_smi == canonical_infered_smi


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_greedy(inferer: MegaMolBARTInference):
    samples = inferer.sample(
        num_samples=3,
        sampling_method="greedy-perturbate",
        scaled_radius=1,
        smis=_SMIS,
    )
    samples = set(samples)
    log.info('\n'.join(_SMIS))
    log.info('\n'.join(samples))
    valid_molecules = []
    for smi in set(samples):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))
    log.info(f'Total samples = {len(samples)} unique samples {len(set(samples))}  valids {len(valid_molecules)}')

    if len(valid_molecules) < len(samples) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_topk(inferer: MegaMolBARTInference):
    samples = inferer.sample(
        num_samples=3,
        sampling_method="topkp-perturbate",
        scaled_radius=0,
        topk=4,
        temperature=2,
        topp=0.0,
        smis=_SMIS,
    )
    nl = "\n"
    log.info(f"INPUTS: \n{nl.join(_SMIS)}\n")
    log.info(f"SAMPLES: \n{nl.join(samples)}\n")
    valid_molecules = []
    for smi in set(samples):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))
    log.info(f'Total samples = {len(samples)} unique samples {len(set(samples))}  valids {len(valid_molecules)}')

    if len(valid_molecules) < len(samples) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_topp(inferer: MegaMolBARTInference):
    samples = inferer.sample(
        num_samples=3,
        sampling_method="topkp-perturbate",
        scaled_radius=0,
        topk=0,
        temperature=2,
        topp=0.9,
        smis=_SMIS,
    )
    # samples = set(samples)
    nl = "\n"
    log.info(f"INPUTS: \n{nl.join(_SMIS)}\n")
    log.info(f"SAMPLES: \n{nl.join(samples)}\n")
    valid_molecules = []
    for smi in set(samples):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))
    log.info(f'Total samples = {len(samples)} unique samples {len(set(samples))}  valids {len(valid_molecules)}')

    if len(valid_molecules) < len(samples) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_beam_search(inferer: MegaMolBARTInference):
    beam_size = 5
    samples = inferer.sample(
        num_samples=beam_size,
        sampling_method="beam-search-perturbate",
        beam_alpha=0,
        smis=_SMIS,
    )
    assert len(samples) == len(_SMIS)
    assert len(samples[0]) == beam_size

    nl = "\n"
    for smi_i, samples_i in zip(_SMIS, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join(samples_i)}\n")

    samples_flat = [item for sublist in samples for item in sublist]
    valid_molecules = []
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))

    log.info(
        f'Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))} '
        f'valids {len(valid_molecules)}'
    )

    if len(valid_molecules) < len(samples_flat) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0
