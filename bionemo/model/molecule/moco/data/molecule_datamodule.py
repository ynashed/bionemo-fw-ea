import pickle
from os.path import join
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn
from torch.utils.data import Subset
from torch_geometric.data import DataLoader, InMemoryDataset


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
        ]


class MoleculeDataModule(LightningDataModule):
    def __init__(self, cfg, only_stats: bool = False):
        super().__init__()
        self.cfg = cfg
        self.datadir = cfg.dataset_root
        root_path = cfg.dataset_root
        self.pin_memory = True

        self.train_dataset = MoleculeDataset(
            split="train",
            root=root_path,
            processed_folder=cfg.processed_folder,
            removed_h=cfg.remove_hs,
            only_stats=only_stats,
        )
        self.val_dataset = MoleculeDataset(
            split="val",
            root=root_path,
            processed_folder=cfg.processed_folder,
            removed_h=cfg.remove_hs,
            only_stats=only_stats,
        )
        self.test_dataset = MoleculeDataset(
            split="test",
            root=root_path,
            processed_folder=cfg.processed_folder,
            removed_h=cfg.remove_hs,
            only_stats=only_stats,
        )

        self.statistics = {
            "train": self.train_dataset.statistics,
            "val": self.val_dataset.statistics,
            "test": self.test_dataset.statistics,
        }

        self.removed_h = cfg.remove_hs

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=False,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=False,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=False,
        )
        return dataloader


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "dataset_root": "/workspace/bionemo/data/pyg_geom_drug",
            "processed_folder": "processed_tiny",
            "batch_size": 4,
            "inference_batch_size": 5,
            "num_workers": 4,
            "remove_hs": False,
            "select_train_subset": 100,
        }
    )

    td = MoleculeDataset(
        split="test",
        root=cfg.dataset_root,
        processed_folder=cfg.processed_folder,
        removed_h=cfg.remove_hs,
        only_stats=False,
    )
    import ipdb

    ipdb.set_trace()
    print(td[0])
    datamodule = MoleculeDataModule(cfg)
    train_dataloader = datamodule.test_dataloader()

    for batch in train_dataloader:
        print(batch)
        break
