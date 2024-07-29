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
from collections import Counter

import numpy as np
import torch
from rdkit import Chem
from torchmetrics import MeanMetric


def normalize(tensor):
    s = tensor.sum()
    assert s > 0
    return tensor / s


def total_variation1d(preds, target):
    """
    Calculate the total variation distance between two 1D tensors.

    Args:
        preds (torch.Tensor): Predictions tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        float: The total variation distance.
        torch.Tensor: The absolute difference between the normalized tensors.
    """
    assert target.dim() == 1 and preds.shape == target.shape, f"preds: {preds.shape}, target: {target.shape}"
    target = normalize(target)
    preds = normalize(preds)
    return torch.sum(torch.abs(preds - target)).item(), torch.abs(preds - target)


def bond_length_distance(molecules, target, bond_types_probabilities):
    generated_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for molecule in molecules:
        cdists = torch.cdist(molecule.positions.unsqueeze(0), molecule.positions.unsqueeze(0)).squeeze(0)
        for bond_type in range(1, 5):
            edges = torch.nonzero(molecule.bond_types == bond_type)
            bond_distances = cdists[edges[:, 0], edges[:, 1]]
            distances_to_consider = torch.round(bond_distances, decimals=2)
            for d in distances_to_consider:
                generated_bond_lenghts[bond_type][d.item()] += 1

    # Normalizing the bond lenghts
    for bond_type in range(1, 5):
        s = sum(generated_bond_lenghts[bond_type].values())
        if s == 0:
            s = 1
        for d, count in generated_bond_lenghts[bond_type].items():
            generated_bond_lenghts[bond_type][d] = count / s

    # Convert both dictionaries to tensors
    min_generated_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in generated_bond_lenghts.values())
    min_target_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in target.values())
    min_length = min(min_generated_length, min_target_length)

    max_generated_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in generated_bond_lenghts.values())
    max_target_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in target.values())
    max_length = max(max_generated_length, max_target_length)

    num_bins = int((max_length - min_length) * 100) + 1
    generated_bond_lengths = torch.zeros(4, num_bins)
    target_bond_lengths = torch.zeros(4, num_bins)

    for bond_type in range(1, 5):
        for d, count in generated_bond_lenghts[bond_type].items():
            bin = int((d - min_length) * 100)
            generated_bond_lengths[bond_type - 1, bin] = count
        for d, count in target[bond_type].items():
            bin = int((d - min_length) * 100)
            target_bond_lengths[bond_type - 1, bin] = count

    cs_generated = torch.cumsum(generated_bond_lengths, dim=1)
    cs_target = torch.cumsum(target_bond_lengths, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 100  # 100 because of bin size
    weighted = w1_per_class * bond_types_probabilities[1:]
    return torch.sum(weighted).item(), w1_per_class


def angle_distance(
    molecules,
    target_angles,
    atom_types_probabilities,
    valencies,
    atom_decoder,
    save_histogram: bool,
):
    num_atom_types = len(atom_types_probabilities)
    generated_angles = torch.zeros(num_atom_types, 180 * 10 + 1)
    for molecule in molecules:
        adj = molecule.bond_types
        pos = molecule.positions
        for atom in range(adj.shape[0]):
            p_a = pos[atom]
            neighbors = torch.nonzero(adj[atom]).squeeze(1)
            for i in range(len(neighbors)):
                p_i = pos[neighbors[i]]
                for j in range(i + 1, len(neighbors)):
                    p_j = pos[neighbors[j]]
                    v1 = p_i - p_a
                    v2 = p_j - p_a
                    assert not torch.isnan(v1).any()
                    assert not torch.isnan(v2).any()
                    prod = torch.dot(v1 / (torch.norm(v1) + 1e-6), v2 / (torch.norm(v2) + 1e-6))
                    if prod > 1:
                        print(
                            f"Invalid angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                            f" {v2 / (torch.norm(v2) + 1e-6)}"
                        )
                    prod.clamp(min=0, max=1)
                    angle = torch.acos(prod)
                    if torch.isnan(angle).any():
                        print(
                            f"Nan obtained in angle {i} {j} -- {prod} -- {v1 / (torch.norm(v1) + 1e-6)} --"
                            f" {v2 / (torch.norm(v2) + 1e-6)}"
                        )
                    else:
                        bin = int(torch.round(angle * 180 / math.pi, decimals=1).item() * 10)
                        generated_angles[molecule.atom_types[atom], bin] += 1

    s = torch.sum(generated_angles, dim=1, keepdim=True)
    s[s == 0] = 1
    generated_angles = generated_angles / s
    if save_histogram:
        np.save("generated_angles_historgram.npy", generated_angles.numpy())

    if type(target_angles) in [np.array, np.ndarray]:
        target_angles = torch.from_numpy(target_angles).float()

    cs_generated = torch.cumsum(generated_angles, dim=1)
    cs_target = torch.cumsum(target_angles, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    # The atoms that have a valency less than 2 should not matter
    valency_weight = torch.zeros(len(w1_per_type), device=w1_per_type.device)
    for i in range(len(w1_per_type)):
        valency_weight[i] = 1 - valencies[atom_decoder[i]][0] - valencies[atom_decoder[i]][1]

    weighted = w1_per_type * atom_types_probabilities * valency_weight
    return (torch.sum(weighted) / (torch.sum(atom_types_probabilities * valency_weight) + 1e-5)).item(), w1_per_type


def dihedral_distance(
    molecules,
    target_dihedrals,
    bond_types_probabilities,
    save_histogram,
):
    def calculate_dihedral_angles(mol):
        def find_dihedrals(mol):
            torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
            torsionQuery = Chem.MolFromSmarts(torsionSmarts)
            matches = mol.GetSubstructMatches(torsionQuery)
            torsionList = []
            btype = []
            for match in matches:
                idx2 = match[0]
                idx3 = match[1]
                bond = mol.GetBondBetweenAtoms(idx2, idx3)
                jAtom = mol.GetAtomWithIdx(idx2)
                kAtom = mol.GetAtomWithIdx(idx3)
                if (
                    (jAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (jAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ) or (
                    (kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (kAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ):
                    continue
                for b1 in jAtom.GetBonds():
                    if b1.GetIdx() == bond.GetIdx():
                        continue
                    idx1 = b1.GetOtherAtomIdx(idx2)
                    for b2 in kAtom.GetBonds():
                        if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                            continue
                        idx4 = b2.GetOtherAtomIdx(idx3)
                        # skip 3-membered rings
                        if idx4 == idx1:
                            continue
                        bt = bond.GetBondTypeAsDouble()
                        # bt = str(bond.GetBondType())
                        # if bond.IsInRing():
                        #     bt += '_R'
                        btype.append(bt)
                        torsionList.append((idx1, idx2, idx3, idx4))
            return np.asarray(torsionList), np.asarray(btype)

        dihedral_idx, dihedral_types = find_dihedrals(mol)

        coords = mol.GetConformer().GetPositions()
        t_angles = []
        for t in dihedral_idx:
            u1, u2, u3, u4 = coords[torch.tensor(t)]

            a1 = u2 - u1
            a2 = u3 - u2
            a3 = u4 - u3

            v1 = np.cross(a1, a2)
            v1 = v1 / (v1 * v1).sum(-1) ** 0.5
            v2 = np.cross(a2, a3)
            v2 = v2 / (v2 * v2).sum(-1) ** 0.5
            porm = np.sign((v1 * a3).sum(-1))
            rad = np.arccos((v1 * v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1) + 1e-9) ** 0.5)
            if not porm == 0:
                rad = rad * porm
            t_angles.append(rad * 180 / torch.pi)

        return np.asarray(t_angles), dihedral_types

    # forget about none and tripple bonds
    bond_types_probabilities[torch.tensor([0, 3])] = 0
    bond_types_probabilities /= bond_types_probabilities.sum()

    num_bond_types = len(bond_types_probabilities)
    generated_dihedrals = torch.zeros(num_bond_types, 180 * 10 + 1)
    for mol in molecules:
        mol = mol.rdkit_mol
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        angles, types = calculate_dihedral_angles(mol)
        # transform types to idx
        types[types == 1.5] = 4
        types = types.astype(int)
        for a, t in zip(np.abs(angles), types):
            if np.isnan(a):
                continue
            generated_dihedrals[t, int(np.round(a, decimals=1) * 10)] += 1

    # normalize
    s = generated_dihedrals.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    generated_dihedrals = generated_dihedrals.float() / s

    if save_histogram:
        np.save("generated_dihedrals_historgram.npy", generated_dihedrals.numpy())

    if type(target_dihedrals) in [np.array, np.ndarray]:
        target_dihedrals = torch.from_numpy(target_dihedrals).float()

    cs_generated = torch.cumsum(generated_dihedrals, dim=1)
    cs_target = torch.cumsum(target_dihedrals, dim=1)

    w1_per_type = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 10

    weighted = w1_per_type * bond_types_probabilities

    return torch.sum(weighted).item(), w1_per_type


class Molecule3DMetrics:
    """
    Class to compute 3D metrics for molecules, including bond lengths, bond angles, and optionally dihedrals.
    """

    def __init__(self, dataset_info, device="cpu", test=False, compute_dihedrals=False):
        self.atom_decoder = (
            dataset_info["atom_decoder"] if isinstance(dataset_info, dict) else dataset_info.atom_decoder
        )
        self.bond_lengths_w1 = MeanMetric().to(device)
        self.angles_w1 = MeanMetric().to(device)
        self.dihedrals_w1 = MeanMetric().to(device) if compute_dihedrals else None
        self.statistics = dataset_info["statistics"]
        self.dataset_info = dataset_info
        self.test = test
        self.compute_dihedrals = compute_dihedrals

    def reset(self):
        """Reset all metrics."""
        for metric in [self.bond_lengths_w1, self.angles_w1]:
            metric.reset()
        if self.compute_dihedrals:
            self.dihedrals_w1.reset()

    def __call__(self, molecules):
        """
        Calculate and return the 3D metrics for a list of molecules.

        Args:
            molecules (list): List of molecule objects.

        Returns:
            dict: Dictionary containing the 3D metrics.
        """
        stat = self.statistics["test"] if self.test else self.statistics["val"]

        bond_lengths_w1, bond_lengths_w1_per_type = bond_length_distance(molecules, stat.bond_lengths, stat.bond_types)
        self.bond_lengths_w1(bond_lengths_w1)
        angles_w1, angles_w1_per_type = angle_distance(
            molecules,
            stat.bond_angles,
            stat.atom_types,
            stat.valencies,
            atom_decoder=self.dataset_info["atom_decoder"],
            save_histogram=self.test,
        )
        self.angles_w1(angles_w1)

        metrics_3d = {
            "bond_lengths": self.bond_lengths_w1.compute().item(),
            "bond_angles": self.angles_w1.compute().item(),
        }

        if self.compute_dihedrals:
            dihedrals_w1, dihedrals_w1_per_type = dihedral_distance(
                molecules,
                stat.dihedrals,
                stat.bond_types,
                save_histogram=self.test,
            )
            self.dihedrals_w1(dihedrals_w1)
            metrics_3d["dihedrals"] = self.dihedrals_w1.compute().item()

        return metrics_3d

    @staticmethod
    def default_values():
        """
        Return default values for the 3D metrics.

        Returns:
            dict: Dictionary containing default 3D metric values.
        """
        return {
            "bond_lengths": 10.0,
            "bond_angles": 10.0,
            "dihedrals": 10.0,
        }
