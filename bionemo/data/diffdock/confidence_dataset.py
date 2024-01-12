# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import math
import os
import pickle
import random
from functools import partial
from typing import Iterable

import numpy as np
import torch
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from bionemo.data.diffdock.confidence_store import ConfidenceStore
from bionemo.data.diffdock.docking_dataset import ProteinLigandDockingDataset, make_cache_path
from bionemo.data.diffdock.heterograph_store import HeterographStore
from bionemo.model.molecule.diffdock.utils.ddp import get_rank
from bionemo.model.molecule.diffdock.utils.diffusion import get_t_schedule
from bionemo.model.molecule.diffdock.utils.diffusion import (
    t_to_sigma as t_to_sigma_compl,
)
from bionemo.model.molecule.diffdock.utils.sampling import randomize_position, sampling


class ListDataset(Dataset):
    def __init__(self, data_list: Iterable):
        super().__init__()
        self.data_list = data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


def get_cache_path(cfg: DictConfig, split: str) -> os.PathLike:
    """Get the full cache path for storing the complex graph data for certain split

    Args:
        cfg (DictConfig): Data config
        split (str): split file name

    Returns:
        os.PathLike: absolute full cache path for this split
    """
    cache_path = cfg.cache_path
    split_path = cfg.split_val if 'val' in split else cfg.get(f"split_{split}")
    full_cache_path = make_cache_path(
        cache_path=cache_path,
        split_path=split_path,
        matching=not cfg.no_torsion,
        all_atoms=cfg.all_atoms,
        limit_complexes=cfg.limit_complexes,
        max_lig_size=cfg.max_lig_size,
        remove_hs=cfg.remove_hs,
        receptor_radius=cfg.receptor_radius,
        c_alpha_max_neighbors=cfg.c_alpha_max_neighbors,
        atom_radius=cfg.atom_radius,
        atom_max_neighbors=cfg.atom_max_neighbors,
        num_conformers=cfg.num_conformers,
        esm_embeddings_path=cfg.esm_embeddings_path,
    )
    return full_cache_path


def get_confidence_cache_path(cache_dir, cache_creation_id=None):
    return os.path.join(
        cache_dir,
        f"confidence_cache_id_{cache_creation_id if cache_creation_id is not None else 'base'}.sqlite3",
    )


class ConfidenceDataset(Dataset):
    """Confidence dataset
    There are receptor-ligand complex graphs, and also ligand poses from reverse diffusion.
    """

    def __init__(
        self,
        cache_path,
        split,
        limit_complexes,
        samples_per_complex,
        all_atoms,
        cfg,
        balance=False,
        use_original_model_cache=True,
        rmsd_classification_cutoff=2,
        cache_ids_to_combine=None,
        cache_creation_id=None,  # set this to anything beside None to enforce caching confidence sqlite
        seed=None,
    ):
        super(ConfidenceDataset, self).__init__()
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.balance = balance
        self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.split = split
        self.cfg = cfg
        self.limit_complexes = limit_complexes
        self.samples_per_complex = samples_per_complex
        self.seed = seed

        self.complex_graphs_cache = (
            self.original_model_cache if self.use_original_model_cache else get_cache_path(cfg, split)
        )
        logging.info(
            "Using the cached complex graphs of the original model config"
            if self.use_original_model_cache
            else "Not using the cached complex graphs of the original model config. "
            + "Instead the complex graphs are used that are at the location given by the dataset "
            + "parameters given to train_confidence.py"
        )
        self.complex_graphs_cache_sqlite_path = os.path.join(self.complex_graphs_cache, "heterographs.sqlite3")

        if not os.path.exists(self.complex_graphs_cache_sqlite_path):
            logging.info(
                f"Complex graphs path does not exist yet: {self.complex_graphs_cache_sqlite_path}."
                + " use build_complex_graphs() to build"
            )
            self.complex_graphs_ready = False
            # self.build_complex_graphs()
        else:
            self.complex_graphs_ready = True

        self.full_cache_path = os.path.join(
            cache_path,
            f"model_{self.cfg.score_model_name}" f"_split_{split}_limit_{limit_complexes}",
        )

        self.confidence_cache_store_path = get_confidence_cache_path(self.full_cache_path, self.cache_creation_id)
        if not os.path.exists(self.confidence_cache_store_path) and self.cache_creation_id is None:
            logging.info(
                f"Confidence dataset does not exist yet: {self.confidence_cache_store_path}."
                + " use build_confidence_dataset() to build"
            )
            self.confidence_dataset_ready = False
            # self.build_confidence_dataset()
        else:
            self.confidence_dataset_ready = True

    def len(self):
        return len(self.dataset_names)

    def build_complex_graphs(self):
        cfg = self.cfg
        split = self.split
        if not os.path.exists(self.complex_graphs_cache_sqlite_path):
            logging.info(f"Create complex graph dataset: {self.complex_graphs_cache_sqlite_path}.")
            complex_dataset = ProteinLigandDockingDataset(
                transform=None,
                root=cfg.data_dir,
                limit_complexes=cfg.limit_complexes,
                receptor_radius=cfg.receptor_radius,
                cache_path=cfg.cache_path,
                split_path=cfg.split_val if 'val' in split else cfg.get(f"split_{split}"),
                remove_hs=cfg.remove_hs,
                max_lig_size=None,
                c_alpha_max_neighbors=cfg.c_alpha_max_neighbors,
                matching=not cfg.no_torsion,
                keep_original=True,
                popsize=cfg.matching_popsize,
                maxiter=cfg.matching_maxiter,
                all_atoms=cfg.all_atoms,
                atom_radius=cfg.atom_radius,
                atom_max_neighbors=cfg.atom_max_neighbors,
                esm_embeddings_path=cfg.esm_embeddings_path,
                require_ligand=True,
                num_workers=cfg.num_workers,
                chunk_size=cfg.chunk_size,
                seed=self.seed,
            )
            complex_dataset.build_complex_graphs()
            self.complex_graphs_ready = True
        else:
            logging.warning(
                "Trying to call build confidence complex graph dataset, "
                f"but cached file is here {self.complex_graphs_cache_sqlite_path}. "
                "skip dataset building, if it is intended, remove the cached file."
            )
            self.complex_graphs_ready = True

    def load_complex_graphs(self):
        local_rank = get_rank()
        if local_rank == 0:
            logging.info(f"Loading complex graphs from: {self.complex_graphs_cache_sqlite_path}")
        self.complex_graphs = HeterographStore(self.complex_graphs_cache_sqlite_path)
        self.complex_graphs_ready = True

    def build_confidence_dataset(self, score_model):
        os.makedirs(self.full_cache_path, exist_ok=True)
        if not os.path.exists(self.confidence_cache_store_path) and self.cache_creation_id is None:
            logging.info(f"Create confidence dataset: {self.confidence_cache_store_path}.")
            self.preprocessing(score_model)
            self.confidence_dataset_ready = True
        else:
            logging.warning(
                "Trying to call build confidence dataset with ligand poses, "
                f"but cached file is here {self.confidence_cache_store_path}. "
                "skip dataset building, if it is intended, remove the cached file."
            )
            self.confidence_dataset_ready = True

    def load_confidence_dataset(self):
        self.load_complex_graphs()

        limit_complexes = self.limit_complexes
        samples_per_complex = self.samples_per_complex
        generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds = (
            [],
            [],
            [],
        )

        if self.cache_ids_to_combine is None:
            logging.info(
                f"Cached RMSDS Found | Loading names, positions and rmsds from: {self.confidence_cache_store_path}"
            )
            confidence_sqlite = ConfidenceStore(self.confidence_cache_store_path)
            generated_rmsd_complex_names = confidence_sqlite[:][0]
            self.full_ligand_positions = confidence_sqlite[:][1]
            self.rmsds = confidence_sqlite[:][2]
            confidence_sqlite.close()
        else:
            for idx, cache_id in enumerate(self.cache_ids_to_combine):
                confidence_cache_id_path = get_confidence_cache_path(self.full_cache_path, cache_id)
                logging.info(
                    f"Merging Cached RMSDS | Loading names, positions and rmsds from cache_id from the path: {confidence_cache_id_path}"
                )
                if not os.path.exists(confidence_cache_id_path):
                    raise Exception(f"The generated ligand positions with cache_id do not exist: {cache_id}")
                confidence_sqlite_id = ConfidenceStore(self.confidence_cache_store_path)
                generated_rmsd_complex_names_id = confidence_sqlite_id[:][0]
                full_ligand_positions_id = confidence_sqlite_id[:][1]
                rmsds_id = confidence_sqlite_id[:][2]
                confidence_sqlite_id.close()

                generated_rmsd_complex_names.extend(generated_rmsd_complex_names_id)
                self.full_ligand_positions.extend(full_ligand_positions_id)
                self.rmsds.extend(rmsds_id)

        logging.info(f"Number of complex graphs: {len(self.complex_graphs)}")
        logging.info(
            "Number of RMSDs and positions for the complex graphs: " f"{len(self.full_ligand_positions)}",
        )

        self.all_samples_per_complex = samples_per_complex * (
            1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine)
        )

        self.positions_rmsds_dict = {
            name: (pos, rmsd)
            for name, pos, rmsd in zip(generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds)
        }

        if os.path.exists(os.path.join(self.complex_graphs_cache, "names.pkl")):
            self.complex_graph_names_dict = pickle.load(
                open(os.path.join(self.complex_graphs_cache, "names.pkl"), "rb")
            )
        else:
            self.complex_graph_names_dict = {self.complex_graphs[k].name: k for k in range(len(self.complex_graphs))}
            pickle.dump(
                self.complex_graph_names_dict, open(os.path.join(self.complex_graphs_cache, "names.pkl"), "wb")
            )

        self.dataset_names = [
            name
            for name in self.positions_rmsds_dict.keys()
            if name in frozenset(self.complex_graph_names_dict.keys())
        ]

        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]

        self.confidence_dataset_ready = True

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graphs[self.complex_graph_names_dict[self.dataset_names[idx]]])
        positions, rmsds = self.positions_rmsds_dict[self.dataset_names[idx]]

        if self.balance:
            if isinstance(self.rmsd_classification_cutoff, ListConfig):
                raise ValueError("a list for --rmsd_classification_cutoff can only be used without --balance")
            label = random.randint(0, 1)
            success = rmsds < self.rmsd_classification_cutoff
            n_success = np.count_nonzero(success)
            if label == 0 and n_success != self.all_samples_per_complex:
                # sample negative complex
                sample = random.randint(0, self.all_samples_per_complex - n_success - 1)
                lig_pos = positions[~success][sample]
                complex_graph["ligand"].pos = torch.from_numpy(lig_pos)
            else:
                # sample positive complex
                if n_success > 0:  # if no successfull sample returns the matched complex
                    sample = random.randint(0, n_success - 1)
                    lig_pos = positions[success][sample]
                    complex_graph["ligand"].pos = torch.from_numpy(lig_pos)
            complex_graph.y = torch.tensor(label).float()
        else:
            sample = random.randint(0, self.all_samples_per_complex - 1)
            complex_graph["ligand"].pos = torch.from_numpy(positions[sample])
            complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
            if isinstance(self.rmsd_classification_cutoff, ListConfig):
                complex_graph.y_binned = torch.tensor(
                    np.logical_and(
                        rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],
                        rmsds[sample] >= [0] + self.rmsd_classification_cutoff,
                    ),
                    dtype=torch.float,
                ).unsqueeze(0)
                complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
            complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()

        complex_graph["ligand"].node_t = {
            "tr": 0 * torch.ones(complex_graph["ligand"].num_nodes),
            "rot": 0 * torch.ones(complex_graph["ligand"].num_nodes),
            "tor": 0 * torch.ones(complex_graph["ligand"].num_nodes),
        }
        complex_graph["receptor"].node_t = {
            "tr": 0 * torch.ones(complex_graph["receptor"].num_nodes),
            "rot": 0 * torch.ones(complex_graph["receptor"].num_nodes),
            "tor": 0 * torch.ones(complex_graph["receptor"].num_nodes),
        }
        if self.all_atoms:
            complex_graph["atom"].node_t = {
                "tr": 0 * torch.ones(complex_graph["atom"].num_nodes),
                "rot": 0 * torch.ones(complex_graph["atom"].num_nodes),
                "tor": 0 * torch.ones(complex_graph["atom"].num_nodes),
            }
        complex_graph.complex_t = {
            "tr": 0 * torch.ones(1),
            "rot": 0 * torch.ones(1),
            "tor": 0 * torch.ones(1),
        }
        return complex_graph

    def preprocessing(self, model):
        # TODO: ideally sampling should be parallel
        logging.info("loading the trained score model for inference to train the confidence model")

        score_model_complex_graphs_cache = get_cache_path(model.cfg.model.get(f"{self.split}_ds"), self.split)
        score_model_complex_graphs = HeterographStore(
            os.path.join(score_model_complex_graphs_cache, "heterographs.sqlite3")
        )
        dataset = ListDataset(score_model_complex_graphs)
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
        )

        t_to_sigma = partial(t_to_sigma_compl, cfg=model.cfg.model)
        tr_schedule = get_t_schedule(denoising_inference_steps=model.cfg.model.denoising_inference_steps)
        rot_schedule = tr_schedule
        tor_schedule = tr_schedule

        logging.info(f"common t (noising time) schedule {tr_schedule}")

        confidence_sqlite = ConfidenceStore(self.confidence_cache_store_path)
        for idx, orig_complex_graph in enumerate(tqdm(loader)):
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            randomize_position(
                data_list,
                model.cfg.model.diffusion.no_torsion,
                False,
                model.cfg.model.diffusion.tr_sigma_max,
            )

            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list is None and failed_convergence_counter <= 5:
                try:
                    predictions_list, confidences = sampling(
                        data_list=data_list,
                        model=model.model.net,
                        denoising_inference_steps=model.cfg.model.denoising_inference_steps,
                        tr_schedule=tr_schedule,
                        rot_schedule=rot_schedule,
                        tor_schedule=tor_schedule,
                        device=model.device,
                        t_to_sigma=t_to_sigma,
                        model_cfg=model.cfg.model,
                        batch_size=10,
                    )
                except Exception as e:
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        logging.warning(
                            f"| WARNING: Sampling failed 5 times for {orig_complex_graph.name[0]} with error {e}"
                        )
            if failed_convergence_counter > 5:
                predictions_list = data_list
            if model.cfg.model.diffusion.no_torsion:
                orig_complex_graph["ligand"].orig_pos = (
                    orig_complex_graph["ligand"].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                )

            filterHs = torch.not_equal(predictions_list[0]["ligand"].x[:, 0], 0).cpu().numpy()

            if isinstance(orig_complex_graph["ligand"].orig_pos, list):
                orig_complex_graph["ligand"].orig_pos = orig_complex_graph["ligand"].orig_pos[0]

            ligand_pos = np.asarray(
                [complex_graph["ligand"].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list]
            )
            orig_ligand_pos = np.expand_dims(
                orig_complex_graph["ligand"].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(),
                axis=0,
            )
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
            assert len(orig_complex_graph.name) == 1

            lig_name = orig_complex_graph.name[0]
            lig_pos = np.asarray([complex_graph["ligand"].pos.cpu().numpy() for complex_graph in predictions_list])
            confidence_sqlite.insert(lig_name, lig_pos, rmsd)
        confidence_sqlite.commit()


def diffdock_confidence_dataset(data_config, mode="train"):
    common_args = {
        "cache_path": data_config.cache_path,
        "samples_per_complex": data_config.samples_per_complex,
        "limit_complexes": data_config.limit_complexes,
        "all_atoms": data_config.all_atoms,
        "balance": data_config.balance,
        "rmsd_classification_cutoff": data_config.rmsd_classification_cutoff,
        "use_original_model_cache": data_config.use_original_model_cache,
        "cache_creation_id": data_config.cache_creation_id,
        "cache_ids_to_combine": data_config.cache_ids_to_combine,
        "seed": data_config.seed,
    }
    dataset = ConfidenceDataset(split=mode, cfg=data_config, **common_args)
    return dataset
