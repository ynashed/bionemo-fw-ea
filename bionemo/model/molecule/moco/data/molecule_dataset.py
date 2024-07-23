# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pickle
from os.path import join
from typing import Optional

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset


full_atom_encoder = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
}

full_atom_decoder = dict(map(reversed, full_atom_encoder.items()))


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Statistics:
    def __init__(
        self,
        num_nodes,
        atom_types,
        bond_types,
        charge_types,
        valencies,
        bond_lengths=None,
        bond_angles=None,
        dihedrals=None,
        is_in_ring=None,
        is_aromatic=None,
        hybridization=None,
        force_norms=None,
    ):
        self.num_nodes = num_nodes
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.dihedrals = dihedrals
        self.is_in_ring = is_in_ring
        self.is_aromatic = is_aromatic
        self.hybridization = hybridization
        self.force_norms = force_norms


def maybe_subset(ds, random_subset: Optional[float] = None, split=None) -> torch.utils.data.Dataset:
    if random_subset is None or split in {"test", "val"}:
        return ds
    else:
        idx = torch.randperm(len(ds))[: int(random_subset * len(ds))]
        return Subset(ds, idx)


def train_subset(dset_len, train_size, seed, filename=None, order=None):
    is_float = isinstance(train_size, float)
    train_size = round(dset_len * train_size) if is_float else train_size

    total = train_size
    assert dset_len >= total, f"The dataset ({dset_len}) is smaller than the " f"combined split sizes ({total})."
    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int64)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]

    if order is not None:
        idx_train = [order[i] for i in idx_train]

    idx_train = np.array(idx_train)

    if filename is not None:
        np.savez(filename, idx_train=idx_train)

    return torch.from_numpy(idx_train)


class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        processed_folder,
        split,
        only_stats=True,
        removed_h=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        assert split in ["train", "val", "test"]
        self.root = root
        self.processed_folder = processed_folder
        self.split = split
        self.removed_h = removed_h
        self.atom_encoder = full_atom_encoder
        if removed_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != "H"}

        super().__init__(root, transform, pre_transform, pre_filter)

        if not only_stats:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = None, None

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
            valencies=load_pickle(self.processed_paths[5]),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[7])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[8])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            bond_lengths=load_pickle(self.processed_paths[10]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[11])).float(),
            dihedrals=torch.from_numpy(np.load(self.processed_paths[12])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[6])

    @property
    def processed_dir(self) -> str:
        return join(self.root, self.processed_folder)

    @property
    def processed_file_names(self):
        h = "noh" if self.removed_h else "h"
        return [
            f"{self.split}_{h}.pt",
            f"{self.split}_n_{h}.pickle",
            f"{self.split}_types_{h}.npy",
            f"{self.split}_bond_types_{h}.npy",
            f"{self.split}_charges_{h}.npy",
            f"{self.split}_valency_{h}.pickle",
            f"{self.split}_smiles.pickle",
            f"{self.split}_is_aromatic_{h}.npy",
            f"{self.split}_is_in_ring_{h}.npy",
            f"{self.split}_hybridization_{h}.npy",
            f"{self.split}_bond_lengths_{h}.pickle",
            f"{self.split}_angles_{h}.npy",
            f"{self.split}_dihedrals_{h}.npy",
        ]
