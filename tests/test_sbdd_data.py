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

import numpy as np
import pytest
from rdkit import Chem
from torch_geometric.transforms import Compose

import bionemo.model.molecule.sbdd.utils.transforms as trans
from bionemo.data.sbdd import reconstruct
from bionemo.data.sbdd.pl_data import ProteinLigandData, torchify_dict
from bionemo.data.sbdd.utils_data import PDBProtein, parse_sdf_file


@pytest.fixture(scope="module")
def protein_ligand_data():
    data_prefix = "examples/tests/test_data/molecule/sbdd"
    lig_paths = ["2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf", "4aaw_A_rec_4ac3_r83_lig_tt_min_0.sdf"]
    prot_paths = ["2z3h_A_rec.pdb", "4aaw_A_rec.pdb"]  #! These are uncut proteins ie the full structure

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom("add_aromatic")
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    transform = Compose(transform_list)

    dataset = []
    for pocket_fn, ligand_fn in zip(prot_paths, lig_paths):
        protein = PDBProtein(os.path.join(data_prefix, pocket_fn))
        ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
        pdb_block_pocket = protein.residues_to_pdb_block(protein.query_residues_ligand(ligand_dict, 10))
        protein = PDBProtein(pdb_block_pocket)
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(protein.to_dict_atom()),
            ligand_dict=torchify_dict(ligand_dict),
        )
        dataset.append(transform(data))

    return dataset


def test_dataset_preprocessing(protein_ligand_data):
    dataset = protein_ligand_data

    assert len(dataset) == 2
    assert dataset[0].protein_element.shape[0] == 409
    assert dataset[0].ligand_smiles == "CN(CC[C@H](N)CC(=O)N[C@H]1CC[C@H](N2C=C[C@@](N)(O)NC2=O)O[C@@H]1C(=O)O)C(=N)N"
    assert dataset[0].ligand_pos.shape[0] == 31


def test_molecule_creation(protein_ligand_data):
    dataset = protein_ligand_data
    for pair in dataset:
        pred_pos, pred_v = pair.ligand_pos.numpy().astype(np.float64), pair.ligand_element.numpy()
        pred_atom_type = [int(x) for x in pred_v]
        pred_aromatic = pair.ligand_atom_feature[:, 2].numpy().astype(np.bool_).tolist()
        mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        smiles = Chem.MolToSmiles(mol)
        assert smiles == Chem.MolToSmiles(Chem.RemoveHs(mol))
