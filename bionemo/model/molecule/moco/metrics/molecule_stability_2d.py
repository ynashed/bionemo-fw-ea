# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import Counter

import torch
from rdkit import Chem
from torchmetrics import MeanMetric

from bionemo.model.molecule.moco.metrics.utils import canonicalize_list


allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},
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


def check_stability(molecule, dataset_info, atom_decoder=None):
    """
    Check the stability of a molecule by verifying its atom types and bond types.

    Args:
        molecule: The molecule to check.
        dataset_info: Dataset information containing atom decoder.
        atom_decoder: Optional atom decoder. If not provided, it will be taken from dataset_info.

    Returns:
        Tuple of torch tensors indicating molecular stability, number of stable bonds, and the total number of bonds.
    """
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


class Molecule2DStability:
    def __init__(self, dataset_info, device="cpu"):
        """
        Initialize the Molecule2DStability class.

        Args:
            dataset_info: Dataset information containing atom decoder.
            device: Device to use for computations.
        """
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_info = dataset_info
        self.atom_stable = MeanMetric().to(device)
        self.mol_stable = MeanMetric().to(device)
        self.validity_metric = MeanMetric().to(device)

    def reset(self):
        """Reset the metrics."""
        for metric in [self.atom_stable, self.mol_stable, self.validity_metric]:
            metric.reset()

    def compute_validity(self, generated):
        """
        Compute the validity of the generated molecules.

        Args:
            generated: List of generated molecules.

        Returns:
            Tuple containing valid SMILES, valid molecules.
        """
        valid_smiles = []
        valid_ids = []
        valid_molecules = []
        num_components = []
        error_message = Counter()
        is_valid = []
        for i, mol in enumerate(generated):
            rdmol = mol.rdkit_mol
            is_valid.append((0, 0))
            if rdmol is not None:
                try:
                    num_atoms = rdmol.GetNumAtoms()
                    bond_type_counter = Counter()
                    # Iterate over all bonds in the molecule and count bond types
                    for bond in rdmol.GetBonds():
                        bond_type = bond.GetBondType()
                        bond_type_counter[bond_type] += 1
                    is_valid[i] = (0, num_atoms)
                    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                    num_components.append(len(mol_frags))
                    print(num_atoms, len(mol_frags), bond_type_counter)
                    # adjacency_matrix = Chem.GetAdjacencyMatrix(rdmol)
                    # print("Adjacency matrix:")
                    # print(adjacency_matrix)
                    # print(rdmol.GetConformer().GetPositions())
                    # print()
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    else:
                        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                        Chem.SanitizeMol(largest_mol)
                        smiles = Chem.MolToSmiles(largest_mol)
                        valid_molecules.append(generated[i])
                        valid_smiles.append(smiles)
                        valid_ids.append(i)
                        is_valid[i] = (1, num_atoms)
                        error_message[5] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
        print(
            f"Error messages: Chem.rdchem.AtomValenceException {error_message[1]}, Chem.rdchem.KekulizeException {error_message[2]}, Chem.rdchem.AtomKekulizeException {error_message[3]}, More than 1 fragment {error_message[4]},"
            f" -- No error {error_message[5]}"
        )
        self.validity_metric.update(value=len(valid_smiles) / len(generated), weight=len(generated))
        # import ipdb; ipdb.set_trace()
        valid_smiles, duplicate_ids = canonicalize_list(valid_smiles)
        valid_molecules = [mol for i, mol in enumerate(valid_molecules) if i not in duplicate_ids]

        return valid_smiles, valid_molecules, is_valid

    def evaluate(self, generated):
        """
        Evaluate the stability and validity of the generated molecules.

        Args:
            generated: List of generated molecules.

        Returns:
            Tuple containing valid SMILES, valid molecules, and validity score.
        """
        valid_smiles, valid_molecules, is_valid = self.compute_validity(generated)
        validity = self.validity_metric.compute().item()
        return valid_smiles, valid_molecules, validity, is_valid

    def __call__(self, molecules):
        """
        Evaluate the generated molecules and return their stability metrics.

        Args:
            molecules: List of generated molecules.

        Returns:
            Dictionary of stability metrics, valid SMILES, valid molecules, and stable molecules.
        """
        stable_molecules = []
        stabs = []
        for mol in molecules:
            mol_stable, at_stable, num_bonds = check_stability(mol, self.dataset_info)
            stabs.append(mol_stable.item())
            self.mol_stable.update(value=mol_stable)
            self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)
            if mol_stable:
                stable_molecules.append(mol)
        valid_smiles, valid_molecules, validity, is_valid = self.evaluate(molecules)
        # import ipdb; ipdb.set_trace()
        both = 0
        A = 0
        B = 0
        C = 0
        for a, b in zip(stabs, is_valid):
            if b[0] and not a:
                print("valid not stable molecule", a, b)
                B += 1
            elif b[0] and a:
                both += 1
            elif not b[0] and a:
                A += 1
            else:
                C += 1
        results = {
            "mol_stable": self.mol_stable.compute().item(),
            "atm_stable": self.atom_stable.compute().item(),
            "validity": validity,
            "stable_valid": both / len(is_valid),
            "not_stable_valid": B / len(is_valid),
            "stable_not_valid": A / len(is_valid),
            "not_stable_not_valid": C / len(is_valid),
        }
        # self.validity_metric.reset()
        # self.mol_stable.reset()
        # self.atom_stable.reset()
        return results, valid_smiles, valid_molecules, stable_molecules, is_valid

    @staticmethod
    def default_values():
        """
        Get default values for the stability metrics.

        Returns:
            Dictionary of default stability metric values.
        """
        return {"mol_stable": 0, "atm_stable": 0, "validity": 0}
