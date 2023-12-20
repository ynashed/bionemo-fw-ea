# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import pickle
from functools import partial

import pytest
import torch
from Bio import SeqIO
from hydra import compose, initialize
from omegaconf import open_dict
from pytorch_lightning import seed_everything

from bionemo.data.diffdock.confidence_dataset import diffdock_confidence_dataset
from bionemo.data.diffdock.confidence_store import ConfidenceStore
from bionemo.data.diffdock.embedding_preprocess import prep_embedding
from bionemo.data.diffdock.embedding_store import EmbeddingStore
from bionemo.data.diffdock.heterograph_store import HeterographStore
from bionemo.data.diffdock.pdbbind import diffdock_build_dataset
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference
from bionemo.model.molecule.diffdock.utils.diffusion import t_to_sigma as t_to_sigma_compl
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, './conf')
TEST_DATA_FILE = os.path.join(THIS_FILE_DIR, './test_data/molecule/diffdock/diffdock_preprocessing_test.zip')
ROOT_DIR = 'diffdock'


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


@pytest.mark.parametrize('config_name', ['diffdock_embedding_test'])
def test_diffdock_embedding_preprocessing(tmp_directory, config_name):
    cfg = get_cfg(tmp_directory, PREPEND_CONFIG_DIR, config_name)

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
@pytest.mark.parametrize('config_name', ['diffdock_score_preprocessing_test'])
def test_diffdock_prepare_score_dataset(tmp_directory, config_name):
    cfg = get_cfg(tmp_directory, PREPEND_CONFIG_DIR, config_name)

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


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.parametrize('config_name', ['diffdock_confidence_preprocessing_test'])
def test_diffdock_prepare_confidence_dataset(tmp_directory, config_name):
    cfg = get_cfg(tmp_directory, PREPEND_CONFIG_DIR, config_name)
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
        assert torch.allclose(torch.from_numpy(sample[1]), torch.from_numpy(ref_sample[1]), atol=2.0)

        # RMSD
        assert torch.allclose(torch.from_numpy(sample[2]), torch.from_numpy(ref_sample[2]), atol=0.2)


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.timeout(40)
@pytest.mark.skip(reason="loading model blocking data preprocessing in diffdock confidence model")
@pytest.mark.parametrize('config_name', ['diffdock_confidence_preprocessing_test'])
def test_diffdock_prepare_confidence_dataset_blocking(tmp_directory, config_name):
    cfg = get_cfg(tmp_directory, PREPEND_CONFIG_DIR, config_name)

    # loading score model will block building complex graphs with multiprocessing
    # comment out the score model loading will make this test pass
    dataset = diffdock_confidence_dataset(cfg.model.train_ds, mode="train")
    score_model = DiffDockModelInference(cfg.score_infer)
    score_model.eval()
    dataset.build_complex_graphs()
