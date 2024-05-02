# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

import torch
from chemprop.features import mol2graph
from rdkit import Chem

from bionemo.model.molecule.dsmbind.fann import FANN, MultiHeadAttention, TransformerEncoderLayer
from bionemo.model.molecule.dsmbind.mpn import MPNEncoder
from bionemo.utils.hydra import load_model_config

from .inference_shared_test_code import get_config_dir


def test_multihead_attention_layer(bionemo_home: Path):
    config_path = get_config_dir(bionemo_home, "dsmbind")
    cfg = load_model_config(config_name="train", config_path=config_path)
    cfg_model = cfg.model
    multihead_attention = MultiHeadAttention(d_model=cfg_model.hidden_size, num_heads=cfg_model.num_heads)
    # Create a random src tensor and a corresponding mask
    seq_length = 10
    src = torch.randn(cfg.train.batch_size, seq_length, cfg_model.hidden_size)
    src_key_padding_mask = torch.ones(cfg.train.batch_size, seq_length).bool()

    # Test the split_head function
    split_output = multihead_attention._split_heads(src, cfg.train.batch_size)
    expected_shape = (
        cfg.train.batch_size,
        cfg_model.num_heads,
        seq_length,
        cfg_model.hidden_size // cfg_model.num_heads,
    )
    assert split_output.shape == expected_shape, "Output tensor shape of the split_head function is incorrect."

    # Test the forward function
    output = multihead_attention(src, src_key_padding_mask=src_key_padding_mask)

    expected_shape = (cfg.train.batch_size, seq_length, cfg_model.hidden_size)
    assert output.shape == expected_shape, "Output tensor shape of Transformer encoder is incorrect."


def test_transformer_encoder_layer(bionemo_home: Path):
    config_path = get_config_dir(bionemo_home, "dsmbind")
    cfg = load_model_config(config_name="train", config_path=config_path)
    cfg_model = cfg.model
    encoder = TransformerEncoderLayer(
        d_model=cfg_model.hidden_size,
        num_heads=cfg_model.num_heads,
        dim_feedforward=cfg_model.hidden_size,
        dropout=cfg_model.dropout,
    )
    # Create a random src tensor and a corresponding mask
    seq_length = 10
    src = torch.randn(cfg.train.batch_size, seq_length, cfg_model.hidden_size)
    src_key_padding_mask = torch.ones(cfg.train.batch_size, seq_length).bool()

    output = encoder(src, src_key_padding_mask=src_key_padding_mask)

    expected_shape = (cfg.train.batch_size, seq_length, cfg_model.hidden_size)
    assert output.shape == expected_shape, "Output tensor shape of Transformer encoder is incorrect."


def test_fann(bionemo_home: Path):
    config_path = get_config_dir(bionemo_home, "dsmbind")
    cfg = load_model_config(config_name="train", config_path=config_path)
    cfg_model = cfg.model
    fann = FANN(cfg_model)
    # Fake input data
    N_max = 42  # max number of ligand atoms in a batch
    M_max = 50  # max number of target residues in a batch
    bind_X = torch.randn(cfg.train.batch_size, N_max, cfg_model.max_residue_atoms, 3)  # ligand atom coordinates
    bind_S = torch.randn(cfg.train.batch_size, N_max, cfg_model.hidden_size)  # ligand atom embeddings
    bind_A = torch.zeros(
        cfg.train.batch_size, N_max, cfg_model.max_residue_atoms, dtype=torch.int64
    )  # ligand atom types
    bind_A[:, :, 1] = 1
    tgt_X = torch.randn(cfg.train.batch_size, M_max, cfg_model.max_residue_atoms, 3)  # target atom coordinates
    tgt_S = torch.zeros(cfg.train.batch_size, M_max, cfg_model.aa_size)
    tgt_S[:, :, 18] = 1  # target residue one-hot embeddings
    tgt_A = torch.ones(
        cfg.train.batch_size, M_max, cfg_model.max_residue_atoms, dtype=torch.int64
    )  # target atom types
    binder = (bind_X, bind_S, bind_A)
    target = (tgt_X, tgt_S, tgt_A)
    output = fann(binder, target)
    expected_shape = (cfg.train.batch_size, N_max + M_max, cfg_model.max_residue_atoms, cfg_model.hidden_size)
    assert output.shape == expected_shape, "Output tensor shape of FANN is incorrect."


def test_mpn(bionemo_home: Path):
    config_path = get_config_dir(bionemo_home, "dsmbind")
    cfg = load_model_config(config_name="train", config_path=config_path)
    cfg_model = cfg.model
    mpn = MPNEncoder(cfg_model)

    smiles_list = ['C(C(=O)O)N', 'c1ccccc1', 'O=C(C)Oc1ccccc1C(=O)O', 'C1CCCCC1']
    molecule_list = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    max_atoms = max(mol.GetNumAtoms() for mol in molecule_list)
    mol_graph_batch = mol2graph(molecule_list)
    output = mpn(mol_graph_batch)
    expected_shape = (len(smiles_list), max_atoms, cfg_model.hidden_size)
    assert output.shape == expected_shape, "Output tensor shape of MPN is incorrect."
