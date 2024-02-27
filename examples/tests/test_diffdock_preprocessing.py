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
import pickle
from functools import partial
from pathlib import Path

import e3nn
import numpy as np
import pytest
import torch
from Bio import SeqIO
from pytorch_lightning import seed_everything

from bionemo.data.diffdock.confidence_dataset import diffdock_confidence_dataset
from bionemo.data.diffdock.confidence_store import ConfidenceStore
from bionemo.data.diffdock.docking_dataset import diffdock_build_dataset
from bionemo.data.diffdock.embedding_preprocess import prep_embedding
from bionemo.data.diffdock.embedding_store import EmbeddingStore
from bionemo.data.diffdock.heterograph_store import HeterographStore
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference
from bionemo.model.molecule.diffdock.utils.diffusion import t_to_sigma as t_to_sigma_compl
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


e3nn.set_optimization_defaults(optimize_einsums=False)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
torch.backends.cudnn.enabled = False


@pytest.fixture(scope='session')
def bionemo_home() -> Path:
    try:
        x = os.environ['BIONEMO_HOME']
    except KeyError:
        raise ValueError("Need to set BIONEMO_HOME in order to run unit tests! See docs for instructions.")
    else:
        yield Path(x).absolute()


@pytest.fixture(scope="session")
def config_path_for_tests(bionemo_home) -> str:
    yield str(bionemo_home / "examples" / "tests" / "conf")


@pytest.mark.needs_gpu
def test_diffdock_embedding_preprocessing(tmp_path, config_path_for_tests):
    cfg = load_model_config(config_name="diffdock_embedding_test", config_path=config_path_for_tests)
    cfg.training_data.output_fasta_file = str(tmp_path / "pdb_sequences.fasta")
    cfg.training_data.esm_embeddings_path = str(tmp_path / "esm2_embeddings.sqlite3")

    prep_embedding(cfg.training_data)
    fasta = {rec.id: str(rec.seq) for rec in SeqIO.parse(cfg.training_data.output_fasta_file, 'fasta')}
    ref_fasta = {rec.id: str(rec.seq) for rec in SeqIO.parse(cfg.ref_fasta_file, 'fasta')}
    for complex_name in ref_fasta.keys():
        assert fasta[complex_name] == ref_fasta[complex_name], f"fasta is incorrect for {complex_name}"

    emb_store = EmbeddingStore(db_path=cfg.training_data.esm_embeddings_path)
    ref_emb_store = EmbeddingStore(db_path=cfg.training_data.esm_embeddings_path)

    for complex_name in ref_fasta.keys():
        embedding = torch.from_numpy(pickle.loads(emb_store.search(complex_name)[0][1]))
        ref_embedding = torch.from_numpy(pickle.loads(ref_emb_store.search(complex_name)[0][1]))

        assert torch.allclose(embedding, ref_embedding)


@pytest.mark.slow
def test_diffdock_prepare_score_dataset(config_path_for_tests, tmp_path):
    cfg = load_model_config(config_name="diffdock_score_preprocessing_test", config_path=config_path_for_tests)
    cfg.model.train_ds.cache_path = str(tmp_path / "data_cache")
    t_to_sigma = partial(t_to_sigma_compl, cfg=cfg.model)

    dataset = diffdock_build_dataset(cfg.model.train_ds, t_to_sigma, _num_conformers=True, mode="train")
    dataset.build_complex_graphs()

    assert os.path.isfile(
        os.path.join(cfg.model.train_ds.cache_path, cfg.train_graph_folder_name, 'heterographs.sqlite3')
    )
    graphs = HeterographStore(
        os.path.join(cfg.model.train_ds.cache_path, cfg.train_graph_folder_name, 'heterographs.sqlite3')
    )
    ref_graphs = HeterographStore(cfg.ref_train_graph_file)
    assert len(graphs) == len(ref_graphs) == 2

    attr_dict = {
        'ligand': ['orig_pos', 'x', 'pos', 'edge_mask', 'mask_rotate'],
        'receptor': ['x', 'pos', 'mu_r_norm', 'side_chain_vecs'],
        ('ligand', 'lig_bond', "ligand"): ['edge_index', 'edge_attr'],
        ('receptor', 'rec_contact', 'receptor'): ['edge_index'],
    }

    for idx in range(2):
        sample = graphs[idx]
        ref_sample = ref_graphs[0] if sample.name == ref_graphs[0].name else ref_graphs[1]

        assert torch.allclose(torch.tensor(sample.rmsd_matching), torch.tensor(ref_sample.rmsd_matching), atol=0.01)
        assert torch.allclose(
            torch.tensor(sample.original_center), torch.tensor(ref_sample.original_center), atol=0.01
        )

        for key in attr_dict:
            for attr in attr_dict[key]:
                assert torch.allclose(
                    torch.tensor(getattr(sample[key], attr)), torch.tensor(getattr(ref_sample[key], attr)), atol=0.01
                ), f"{key}.{attr} is wrong"


@pytest.mark.needs_fork
@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
def test_diffdock_prepare_confidence_dataset(tmp_path, config_path_for_tests):
    cfg = load_model_config(config_name="diffdock_confidence_preprocessing_test", config_path=config_path_for_tests)
    cfg.model.train_ds.cache_path = str(tmp_path / "data_cache")
    seed_everything(cfg.seed)

    dataset = diffdock_confidence_dataset(cfg.model.train_ds, mode="train")
    dataset.build_complex_graphs()

    assert os.path.isfile(
        os.path.join(cfg.model.train_ds.cache_path, cfg.train_graph_folder_name, 'heterographs.sqlite3')
    )
    graphs = HeterographStore(
        os.path.join(cfg.model.train_ds.cache_path, cfg.train_graph_folder_name, 'heterographs.sqlite3')
    )
    ref_graphs = HeterographStore(cfg.ref_train_graph_file)
    assert len(graphs) == len(ref_graphs) == 2

    attr_dict = {
        'ligand': ['orig_pos', 'x', 'pos', 'edge_mask', 'mask_rotate'],
        'receptor': ['x', 'pos', 'mu_r_norm', 'side_chain_vecs'],
        'atom': ['x', 'pos'],
        ('ligand', 'lig_bond', "ligand"): ['edge_index', 'edge_attr'],
        ('receptor', 'rec_contact', 'receptor'): ['edge_index'],
        ('atom', 'atom_contact', 'atom'): ['edge_index'],
        ('atom', 'atom_rec_contact', 'receptor'): ['edge_index'],
    }

    for idx in range(2):
        sample = graphs[idx]
        ref_sample = ref_graphs[0] if sample.name == ref_graphs[0].name else ref_graphs[1]

        assert torch.allclose(torch.tensor(sample.rmsd_matching), torch.tensor(ref_sample.rmsd_matching), atol=0.01)
        assert torch.allclose(
            torch.tensor(sample.original_center), torch.tensor(ref_sample.original_center), atol=0.01
        )

        for key in attr_dict:
            for attr in attr_dict[key]:
                assert torch.allclose(
                    torch.tensor(getattr(sample[key], attr)), torch.tensor(getattr(ref_sample[key], attr)), atol=0.01
                ), f"{key}.{attr} is wrong"

    score_model = DiffDockModelInference(cfg.score_infer)
    score_model.eval()
    dataset.build_confidence_dataset(score_model)
    teardown_apex_megatron_cuda()

    assert os.path.isfile(
        os.path.join(cfg.model.train_ds.cache_path, cfg.ligand_pose_folder_name, 'confidence_cache_id_base.sqlite3')
    )
    ligand_poses = ConfidenceStore(
        os.path.join(cfg.model.train_ds.cache_path, cfg.ligand_pose_folder_name, 'confidence_cache_id_base.sqlite3')
    )
    ref_ligand_poses = ConfidenceStore(cfg.ref_ligand_pose_file)

    assert len(ligand_poses) == len(ref_ligand_poses) == 2

    for idx in range(2):
        sample = ligand_poses[idx]
        ref_sample = ref_ligand_poses[0] if sample[0] == ref_ligand_poses[0][0] else ref_ligand_poses[1]
        # ligand positions
        assert np.allclose(sample[1], ref_sample[1], atol=0.01)

        # RMSD
        assert np.allclose(sample[2], ref_sample[2], atol=0.01)
