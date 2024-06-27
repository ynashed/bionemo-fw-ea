# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Geometry import Point3D
from torchmetrics import MeanMetric


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}

_fscores = None


def check_stability(molecule, dataset_info, atom_decoder=None):
    device = molecule.atom_types.device
    if atom_decoder is None:
        atom_decoder = dataset_info["atom_decoder"] if isinstance(dataset_info, dict) else dataset_info.atom_decoder
    atom_types = molecule.atom_types
    edge_types = molecule.bond_types

    edge_types[edge_types == 4] = 1.5
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    for atom_type, valency, charge in zip(atom_types, valencies, molecule.charges):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if isinstance(possible_bonds, int):
            is_stable = possible_bonds == valency
        elif isinstance(possible_bonds, dict):
            expected_bonds = possible_bonds.get(charge, possible_bonds[0])
            is_stable = valency in (expected_bonds if isinstance(expected_bonds, list) else [expected_bonds])
        else:
            is_stable = valency in possible_bonds
        if not is_stable:
            mol_stable = False
        n_stable_bonds += int(is_stable)

    return (
        torch.tensor([mol_stable], dtype=torch.float, device=device),
        torch.tensor([n_stable_bonds], dtype=torch.float, device=device),
        len(atom_types),
    )


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def readFragmentScores(name="fpscores"):
    import gzip
    import os.path as op
    import pickle

    global _fscores
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open(f"{name}.pkl.gz"))
    _fscores = {i[j]: float(i[0]) for i in data for j in range(1, len(i))}


def get_fingerprints_from_smileslist(smiles_list):
    return [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048) for smiles in smiles_list]


def canonicalize(smiles, include_stereocenters=True, remove_hs=False):
    mol = Chem.MolFromSmiles(smiles)
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters) if mol else None


def canonicalize_list(smiles_list, include_stereocenters=True, remove_hs=False):
    canonicalized_smiles = [canonicalize(smiles, include_stereocenters, remove_hs) for smiles in smiles_list]
    canonicalized_smiles = [s for s in canonicalized_smiles if s]

    unique_smiles = []
    seen = set()
    duplicate_ids = []

    for i, smiles in enumerate(canonicalized_smiles):
        if smiles not in seen:
            unique_smiles.append(smiles)
            seen.add(smiles)
        else:
            duplicate_ids.append(i)

    return unique_smiles, duplicate_ids


def analyze_stability_for_molecules(molecule_list, dataset_info, smiles_train, device="cpu"):
    metrics = BasicMolecularMetrics(dataset_info, smiles_train=smiles_train, device=device)
    return metrics(molecule_list)


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = sum(_fscores.get(bitId, -4) * v for bitId, v in fps.items()) / sum(fps.values())
    sizePenalty = m.GetNumAtoms() ** 1.005 - m.GetNumAtoms()
    stereoPenalty = math.log10(len(Chem.FindMolChiralCenters(m, includeUnassigned=True)) + 1)
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = math.log10(2) if any(len(r) > 8 for r in m.GetRingInfo().AtomRings()) else 0
    score2 = 0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
    score3 = math.log(m.GetNumAtoms() / len(fps)) * 0.5 if m.GetNumAtoms() > len(fps) else 0
    sascore = score1 + score2 + score3
    sascore = 11.0 - (sascore + 5) / 7.5 * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    return min(max(sascore, 1.0), 10.0)


class BasicMolecularMetrics:
    def __init__(self, dataset_info, device="cpu"):
        self.atom_decoder = (
            dataset_info["atom_decoder"] if isinstance(dataset_info, dict) else dataset_info.atom_decoder
        )
        self.dataset_info = dataset_info
        self.atom_stable = MeanMetric().to(device)
        self.mol_stable = MeanMetric().to(device)
        self.validity_metric = MeanMetric().to(device)

    def reset(self):
        for metric in [self.atom_stable, self.mol_stable, self.validity_metric]:
            metric.reset()

    def compute_validity(self, generated):
        valid_smiles, valid_molecules = [], []
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                    if len(mol_frags) == 1:
                        Chem.SanitizeMol(mol_frags[0])
                        smiles = Chem.MolToSmiles(mol_frags[0])
                        valid_molecules.append(mol)
                        valid_smiles.append(smiles)
                except (
                    Chem.rdchem.AtomValenceException,
                    Chem.rdchem.KekulizeException,
                    Chem.rdchem.AtomKekulizeException,
                    ValueError,
                ) as e:
                    print(f"Error sanitizing molecule: {e}")

        self.validity_metric.update(value=len(valid_smiles) / len(generated), weight=len(generated))
        valid_smiles, duplicate_ids = canonicalize_list(valid_smiles)
        valid_molecules = [mol for i, mol in enumerate(valid_molecules) if i not in duplicate_ids]
        return valid_smiles, valid_molecules

    def evaluate(self, generated):
        valid_smiles, valid_molecules = self.compute_validity(generated)
        validity = self.validity_metric.compute()
        return valid_smiles, valid_molecules, validity.item()

    def __call__(self, molecules):
        stable_molecules = []
        for mol in molecules:
            mol_stable, at_stable, num_bonds = check_stability(mol, self.dataset_info)
            self.mol_stable.update(value=mol_stable)
            self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)
            if mol_stable:
                stable_molecules.append(mol)
        stability_dict = {
            "mol_stable": self.mol_stable.compute().item(),
            "atm_stable": self.atom_stable.compute().item(),
        }
        valid_smiles, valid_molecules, validity = self.evaluate(molecules)
        validity_dict = {"validity": validity}

        if valid_molecules:
            mols = [mol.rdkit_mol for mol in valid_molecules]
            qed_scores = [self.calculate_qed(mol) for mol in mols]
            sa_scores = [self.calculate_sa(mol) for mol in mols]
            logp_scores = [self.calculate_logp(mol) for mol in mols]
            lipinski_scores = [self.calculate_lipinski(mol) for mol in mols]

            statistics_dict = {
                "QED": np.mean(qed_scores),
                "SA": np.mean(sa_scores),
                "LogP": np.mean(logp_scores),
                "Lipinski": np.mean(lipinski_scores),
            }
        else:
            statistics_dict = {"QED": -1.0, "SA": -1.0, "LogP": -1.0, "Lipinski": -1.0}

        return stability_dict, validity_dict, statistics_dict, valid_smiles, stable_molecules, valid_molecules

    def calculate_qed(self, rdmol):
        return QED.qed(rdmol)

    def calculate_sa(self, rdmol):
        sa = calculateScore(rdmol)
        return round((10 - sa) / 9, 2)

    def calculate_logp(self, rdmol):
        return Crippen.MolLogP(rdmol)

    def calculate_lipinski(self, rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        logp = Crippen.MolLogP(rdmol)
        rule_4 = -2 <= logp <= 5
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return sum(int(rule) for rule in [rule_1, rule_2, rule_3, rule_4, rule_5])


class Molecule:
    def __init__(self, atom_types, bond_types, positions, dataset_info, charges=None):
        self.atom_types = torch.tensor(atom_types, dtype=torch.long)
        self.bond_types = torch.tensor(bond_types, dtype=torch.long)
        self.positions = torch.tensor(positions, dtype=torch.float64)  # Ensure positions are float64
        self.charges = (
            torch.tensor(charges, dtype=torch.long)
            if charges is not None
            else torch.zeros(len(atom_types), dtype=torch.long)
        )
        self.dataset_info = dataset_info
        self.rdkit_mol = self.build_rdkit_mol()

    def build_rdkit_mol(self):
        mol = Chem.RWMol()
        atom_decoder = self.dataset_info["atom_decoder"]
        for i, atom_type in enumerate(self.atom_types):
            try:
                atom = Chem.Atom(atom_decoder[atom_type.item()])
                atom.SetFormalCharge(self.charges[i].item())
                mol.AddAtom(atom)
            except Exception as e:
                print(f"Error adding atom: {e}")

        bond_types = torch.triu(self.bond_types, diagonal=1)
        bond_dict = [
            None,
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        for i in range(bond_types.size(0)):
            for j in range(i + 1, bond_types.size(1)):
                bond_type = bond_types[i, j].item()
                if bond_type > 0:
                    try:
                        mol.AddBond(i, j, bond_dict[bond_type])
                    except Exception as e:
                        print(f"Error adding bond: {e}")

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            try:
                pos = self.positions[i].numpy()
                conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
            except Exception as e:
                print(f"Error setting atom position: {e}")
        mol.AddConformer(conf)

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
        return mol


def get_molecules(mol_dict, dataset_info):
    batch = mol_dict["batch"]
    edge_index = mol_dict["edge_index"]
    edge_attr = mol_dict["edge_attr"]
    h = mol_dict["h"]
    x = mol_dict["x"]

    molecule_list = []
    for idx in torch.unique(batch):
        idx_mask = batch == idx
        atom_types = h[idx_mask]
        positions = x[idx_mask]

        # Create bond matrix
        edge_mask = (edge_index[0] >= idx_mask.nonzero(as_tuple=True)[0].min()) & (
            edge_index[0] <= idx_mask.nonzero(as_tuple=True)[0].max()
        )
        bonds = edge_attr[edge_mask]
        bond_indices = edge_index[:, edge_mask]

        # Adjust bond indices to local molecule
        local_bond_indices = bond_indices - bond_indices[0].min()

        bond_matrix = torch.zeros((len(atom_types), len(atom_types)), dtype=torch.long)
        for src, dst, bond in zip(local_bond_indices[0], local_bond_indices[1], bonds):
            bond_matrix[src, dst] = bond
            bond_matrix[dst, src] = bond

        molecule = Molecule(
            atom_types=atom_types.tolist(),
            bond_types=bond_matrix.tolist(),
            positions=positions.tolist(),
            charges=None,
            dataset_info=dataset_info,
        )
        molecule_list.append(molecule)

    return molecule_list


def test_get_molecule():
    dataset_info = {
        "atom_decoder": {0: "H", 1: "C", 2: "N", 3: "O"},
    }

    data = {
        'batch': torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        'edge_attr': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        'edge_index': torch.tensor(
            [[0, 1, 1, 3, 4, 4, 6, 7, 7, 0, 2, 1], [1, 0, 2, 4, 3, 5, 7, 6, 8, 2, 0, 2]], dtype=torch.long
        ),
        'h': torch.tensor([0, 1, 2, 1, 2, 3, 0, 1, 2], dtype=torch.long),
        'x': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float64,
        ),
    }

    molecules = get_molecules(data, dataset_info)

    # Print molecule info for verification
    for i, mol in enumerate(molecules):
        print(f"Molecule {i}:")
        print(f"  Atom types: {mol.atom_types}")
        print(f"  Bond types:\n{mol.bond_types}")
        print(f"  Positions:\n{mol.positions}")
        print(f"  Charges: {mol.charges}")
        print()


def test_basic_molecular_metrics():
    dataset_info = {
        "atom_decoder": {0: "H", 1: "C", 2: "N", 3: "O"},
    }

    molecules = [
        # H2O
        Molecule(
            atom_types=[3, 0, 0],
            bond_types=[[0, 1, 1], [1, 0, 0], [1, 0, 0]],
            positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2396, 0.9266, 0.0]],
            charges=[0, 0, 0],
            dataset_info=dataset_info,
        ),
        # H2O invalid
        Molecule(
            atom_types=[3, 0, 0],
            bond_types=[[0, 1, 2], [1, 0, 0], [1, 0, 0]],
            positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2396, 0.9266, 0.0]],
            charges=[0, 0, 0],
            dataset_info=dataset_info,
        ),
        # CO2
        Molecule(
            atom_types=[1, 3, 3],
            bond_types=[[0, 2, 2], [2, 0, 0], [2, 0, 0]],
            positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]],
            charges=[0, 0, 0],
            dataset_info=dataset_info,
        ),
        # C2H5OH (Ethanol)
        Molecule(
            atom_types=[1, 1, 3, 0, 0, 0, 0, 0, 0],
            bond_types=[
                [0, 1, 0, 1, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
            ],
            positions=[
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [2.54, 0.0, 0.0],
                [3.03, 1.21, 0.0],
                [-0.63, -0.63, 0.0],
                [-0.63, 0.63, 0.0],
                [1.09, -0.63, -0.63],
                [1.09, -0.63, 0.63],
                [2.54, -0.63, 0.0],
            ],
            charges=[0, 0, 0, 0, 0, 0, 0, 0, 0],
            dataset_info=dataset_info,
        ),
        # CO2
        Molecule(
            atom_types=[1, 3, 3],
            bond_types=[[0, 2, 2], [2, 0, 0], [2, 0, 0]],
            positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]],
            charges=None,
            dataset_info=dataset_info,
        ),
    ]

    metrics = BasicMolecularMetrics(dataset_info)
    stability_dict, validity_dict, statistics_dict, valid_smiles, stable_molecules, valid_molecules = metrics(
        molecules
    )

    print("Stability Dict:", stability_dict)
    print("Validity Dict:", validity_dict)
    print("Statistics Dict:", statistics_dict)
    print("Valid Smiles:", valid_smiles)
    print("Stable Molecules:", stable_molecules)
    print("Valid Molecules:", valid_molecules)


if __name__ == "__main__":
    test_get_molecule()
    test_basic_molecular_metrics()
