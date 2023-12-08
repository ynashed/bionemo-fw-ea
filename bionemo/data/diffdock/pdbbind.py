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

import binascii
import copy
import math
import multiprocessing
import os
import pickle
import random
from collections import defaultdict
from concurrent import futures
from contextlib import contextmanager
from functools import lru_cache
from multiprocessing import Pool, get_start_method, set_start_method
from typing import Literal, Optional

import numpy as np
import torch
from nemo.utils import logging
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform

from bionemo.data.diffdock.embedding_store import EmbeddingStore
from bionemo.data.diffdock.heterograph_store import HeterographStore
from bionemo.data.diffdock.process_mols import (
    extract_receptor_structure,
    get_lig_graph_with_matching,
    get_rec_graph,
    parse_pdb_from_path,
    parse_receptor,
    read_molecule,
)
from bionemo.model.molecule.diffdock.utils import so3, torus
from bionemo.model.molecule.diffdock.utils.ddp import get_rank
from bionemo.model.molecule.diffdock.utils.diffusion import modify_conformer, set_time


def make_cache_path(
    cache_path="data/data_cache",
    split_path="data/",
    matching=True,
    protein_path_list=None,
    ligand_descriptions=None,
    all_atoms=False,
    limit_complexes=0,
    max_lig_size=None,
    remove_hs=False,
    receptor_radius=30,
    c_alpha_max_neighbors=None,
    atom_radius=5,
    atom_max_neighbors=None,
    num_conformers=1,
    esm_embeddings_path=None,
    keep_local_structures=False,
):
    if matching or protein_path_list is not None and ligand_descriptions is not None:
        cache_prefix = "torsion"
    if all_atoms:
        cache_prefix = "allatoms"
    full_cache_path = os.path.join(
        cache_path,
        f"{cache_prefix}_limit{limit_complexes}"
        f"_INDEX{os.path.splitext(os.path.basename(split_path))[0]}"
        f"_maxLigSize{max_lig_size}_H{int(not remove_hs)}"
        f"_recRad{receptor_radius}_recMax{c_alpha_max_neighbors}"
        + ("" if not all_atoms else f"_atomRad{atom_radius}_atomMax{atom_max_neighbors}")
        + ("" if not matching or num_conformers == 1 else f"_confs{num_conformers}")
        + ("" if esm_embeddings_path is None else "_esmEmbeddings")
        + ("" if not keep_local_structures else "_keptLocalStruct")
        + (
            ""
            if protein_path_list is None or ligand_descriptions is None
            else str(binascii.crc32("".join(ligand_descriptions + protein_path_list).encode()))
        ),
    )
    return full_cache_path


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom

    def __call__(self, data):
        t = np.random.uniform()
        t_tr, t_rot, t_tor = t, t, t
        return self.apply_noise(data, t_tr, t_rot, t_tor)

    def apply_noise(self, data, t_tr, t_rot, t_tor, tr_update=None, rot_update=None, torsion_updates=None):
        if not torch.is_tensor(data["ligand"].pos):
            data["ligand"].pos = random.choice(data["ligand"].pos)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
        set_time(data, t_tr, t_rot, t_tor, 1, self.all_atom, device=None)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = (
            np.random.normal(loc=0.0, scale=tor_sigma, size=data["ligand"].edge_mask.sum())
            if torsion_updates is None
            else torsion_updates
        )
        torsion_updates = None if self.no_torsion else torsion_updates
        modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)

        data.tr_score = -tr_update / tr_sigma**2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data["ligand"].edge_mask.sum()) * tor_sigma
        return data


@contextmanager
def ForkingBehavior(
    *,
    start_method: Literal['spawn', 'fork', 'forkserver'],
    force: bool,
):
    """Contextmanager to set the method for starting child processes in multiprocessing.
    Refer to https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method

    Args:
        start_method (Literal['spawn', 'fork', 'forkserver']): multiprocessing start method to use in the context
        force (bool): Raises RuntimeError if the start method has already been set and force is not True.
                      If start_method is None and force is True then the start method is set to None.
                      If start_method is None and force is False then the context is set to the default context.
    """
    prev_start_method = get_start_method()
    set_start_method(start_method, force=force)
    try:
        yield
    finally:
        set_start_method(prev_start_method, force=True)


class PDBBind(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        cache_path="/data/data_cache",
        split_path="/data/",
        limit_complexes=0,
        receptor_radius=30,
        num_workers=1,
        c_alpha_max_neighbors=None,
        popsize=15,
        maxiter=15,
        matching=True,
        keep_original=False,
        max_lig_size=None,
        remove_hs=False,
        num_conformers=1,
        all_atoms=False,
        atom_radius=5,
        atom_max_neighbors=None,
        esm_embeddings_path=None,
        require_ligand=False,
        protein_path_list=None,
        ligand_descriptions=None,
        keep_local_structures=False,
        chunk_size=5,
    ):
        super(PDBBind, self).__init__(root, transform)
        self.pdbbind_dir = root
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.chunk_size = chunk_size

        self.heterograph_store: Optional[HeterographStore] = None
        self.full_cache_path = self.heterograph_cache_path(cache_path)
        if not os.path.exists(os.path.join(self.full_cache_path, "heterographs.sqlite3")):
            self.complex_graphs_ready = False
            # self.build_complex_graphs():
        else:
            self.complex_graphs_ready = True

    def build_complex_graphs(self):
        local_rank = get_rank()
        if local_rank == 0:
            os.makedirs(self.full_cache_path, exist_ok=True)
        if not os.path.exists(os.path.join(self.full_cache_path, "heterographs.sqlite3")):
            if self.protein_path_list is None or self.ligand_descriptions is None:
                self.preprocessing()
                self.complex_graphs_ready = True
        else:
            logging.warning(
                "Trying to call build scorec complex graph dataset, "
                f"but cached file is here {self.confidence_cache_store_path}. "
                "skip dataset building, if it is intended, remove the cached file."
            )
            self.complex_graphs_ready = True

    def load_complex_graphs(self):
        if self.complex_graphs_ready:
            self.heterograph_store = HeterographStore(os.path.join(self.full_cache_path, "heterographs.sqlite3"))
        else:
            raise RuntimeError(
                f"Failed to load cached heterographs.sqlite3 file in this folder {self.full_cache_path}"
            )

    def heterograph_cache_path(self, cache_path):
        full_cache_path = make_cache_path(
            cache_path=cache_path,
            split_path=self.split_path,
            matching=self.matching,
            protein_path_list=self.protein_path_list,
            ligand_descriptions=self.ligand_descriptions,
            all_atoms=self.all_atoms,
            limit_complexes=self.limit_complexes,
            max_lig_size=self.max_lig_size,
            remove_hs=self.remove_hs,
            receptor_radius=self.receptor_radius,
            c_alpha_max_neighbors=self.c_alpha_max_neighbors,
            atom_radius=self.atom_radius,
            atom_max_neighbors=self.atom_max_neighbors,
            num_conformers=self.num_conformers,
            esm_embeddings_path=self.esm_embeddings_path,
            keep_local_structures=self.keep_local_structures,
        )
        return full_cache_path

    @lru_cache(maxsize=None)
    def len(self):
        return len(self.heterograph_store)

    def get(self, idx):
        complex_graph = self.heterograph_store[idx]
        return complex_graph

    def preprocessing(self):
        with ForkingBehavior(start_method="spawn", force=True):
            logging.info(
                f"[preprocessing] processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]"
            )
            logging.info(f"[preprocessing] reading complexes from split file {self.split_path}")
            complex_names_all = read_strings_from_txt(self.split_path)
            if self.limit_complexes is not None and self.limit_complexes != 0:
                complex_names_all = complex_names_all[: self.limit_complexes]

            num_cores = multiprocessing.cpu_count()
            if self.num_workers < num_cores:
                logging.info(f"num_workers < num_cores: {self.num_workers} < {num_cores}")

            if self.esm_embeddings_path is not None:
                logging.info(
                    f"[preprocessing] loading {len(complex_names_all)} complexes with {self.num_workers} threads."
                )
                chain_embeddings_dictlist = defaultdict(list)

                def _pattern_search(complex_name):
                    emb_store = EmbeddingStore(db_path=self.esm_embeddings_path)
                    result = emb_store.search(complex_name)
                    result = [pickle.loads(x[1]) for x in result]
                    chain_embeddings_dictlist[complex_name] = result
                    emb_store.conn.close()

                with futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    executor.map(_pattern_search, complex_names_all)
                lm_embeddings_chains_all = []
                for name in complex_names_all:
                    lm_embeddings_chains_all.append(chain_embeddings_dictlist[name])
            else:
                lm_embeddings_chains_all = [None] * len(complex_names_all)

            chunk_size = len(complex_names_all) // (self.num_workers * self.chunk_size)
            chunks = math.ceil(len(complex_names_all) / chunk_size)
            complex_chunks = [complex_names_all[chunk_size * i : chunk_size * (i + 1)] for i in range(chunks)]
            lm_chunks = [lm_embeddings_chains_all[chunk_size * i : chunk_size * (i + 1)] for i in range(chunks)]

            with Pool(self.num_workers) as p:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                logging.info(f"Computing for {len(complex_names_all)} complexes...")
                total = 0
                for lig_cnt, _ in map_fn(self.get_complex, zip(complex_chunks, lm_chunks)):
                    total += lig_cnt
                logging.info(f"Total processed complexes: {total}")

    def get_complex(self, par):
        names, lm_embedding_chains = par
        ligands = None
        ligand_descriptions = None

        total_complexes = 0
        total_ligands = 0
        for i in range(len(names)):
            name = names[i]
            lm_embedding_chain = lm_embedding_chains[i]
            ligand = None
            ligand_description = None

            # TODO: Are these two variables necessary?
            if ligands and ligand_descriptions:
                ligand = ligands[i]
                ligand_description = ligand_descriptions[i]

            if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
                logging.warning(f"Folder not found: {name}")
                return [], []

            if ligand is not None:
                rec_model = parse_pdb_from_path(name)
                name = f"{name}____{ligand_description}"
                ligs = [ligand]
            else:
                try:
                    rec_model = parse_receptor(name, self.pdbbind_dir)
                except Exception as e:
                    logging.error(f"Skipping {name} because of the error:")
                    logging.error(e)
                    return [], []

                ligs = read_mols(self.pdbbind_dir, name, remove_hs=False)
            complex_graphs = []
            failed_indices = []
            for i, lig in enumerate(ligs):
                if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                    logging.warning(
                        f"Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data."
                    )
                    continue
                complex_graph = HeteroData()
                complex_graph["name"] = name
                try:
                    get_lig_graph_with_matching(
                        lig,
                        complex_graph,
                        self.popsize,
                        self.maxiter,
                        self.matching,
                        self.keep_original,
                        self.num_conformers,
                        remove_hs=self.remove_hs,
                    )
                    rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(
                        copy.deepcopy(rec_model), lig, lm_embedding_chains=lm_embedding_chain
                    )
                    if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                        logging.warning(
                            f"LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}."
                        )
                        failed_indices.append(i)
                        continue

                    get_rec_graph(
                        rec,
                        rec_coords,
                        c_alpha_coords,
                        n_coords,
                        c_coords,
                        complex_graph,
                        rec_radius=self.receptor_radius,
                        c_alpha_max_neighbors=self.c_alpha_max_neighbors,
                        all_atoms=self.all_atoms,
                        atom_radius=self.atom_radius,
                        atom_max_neighbors=self.atom_max_neighbors,
                        remove_hs=self.remove_hs,
                        lm_embeddings=lm_embeddings,
                    )

                except Exception as e:
                    logging.error(f"Skipping {name} because of the error: {e}")
                    failed_indices.append(i)
                    continue

                protein_center = torch.mean(complex_graph["receptor"].pos, dim=0, keepdim=True)
                complex_graph["receptor"].pos -= protein_center
                if self.all_atoms:
                    complex_graph["atom"].pos -= protein_center

                if (not self.matching) or self.num_conformers == 1:
                    complex_graph["ligand"].pos -= protein_center
                else:
                    for p in complex_graph["ligand"].pos:
                        p -= protein_center

                complex_graph.original_center = protein_center
                complex_graphs.append(complex_graph)
            for idx_to_delete in sorted(failed_indices, reverse=True):
                del ligs[idx_to_delete]

            while True:
                try:
                    hetero_store = HeterographStore(os.path.join(self.full_cache_path, "heterographs.sqlite3"))
                    for lig, complex_graph in zip(ligs, complex_graphs):
                        hetero_store.insert(lig, complex_graph)
                    hetero_store.commit()
                    hetero_store.conn.close()
                    break
                except Exception as e:
                    logging.warning(f"Retrying to commit to sqlite because error: {e}")

            total_complexes += len(complex_graphs)
            total_ligands += len(ligs)
        return total_ligands, total_complexes


def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f"{name}_ligand.sdf"), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, name, f"{name}_ligand.mol2"), remove_hs=remove_hs, sanitize=True)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and "rdkit" not in file:
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            # read mol2 file if sdf file cannot be sanitized
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):
                logging.warning(
                    "Using the .sdf file failed. We found a .mol2 file instead and are trying to use that."
                )
                lig = read_molecule(
                    os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True
                )
            if lig is not None:
                ligs.append(lig)
    return ligs


def diffdock_build_dataset(data_config, t_to_sigma, _num_conformers=True, mode="train"):
    transform = NoiseTransform(
        t_to_sigma=t_to_sigma, no_torsion=data_config.no_torsion, all_atom=data_config.all_atoms
    )
    common_args = {
        "transform": transform,
        "root": data_config.data_dir,
        "limit_complexes": data_config.limit_complexes,
        "receptor_radius": data_config.receptor_radius,
        "c_alpha_max_neighbors": data_config.c_alpha_max_neighbors,
        "remove_hs": data_config.remove_hs,
        "max_lig_size": data_config.max_lig_size,
        "matching": not data_config.no_torsion,
        "popsize": data_config.matching_popsize,
        "maxiter": data_config.matching_maxiter,
        "num_workers": data_config.num_workers,
        "all_atoms": data_config.all_atoms,
        "atom_radius": data_config.atom_radius,
        "atom_max_neighbors": data_config.atom_max_neighbors,
        "esm_embeddings_path": data_config.esm_embeddings_path,
        "chunk_size": data_config.get("chunk_size", 5),
    }
    if mode == "train":
        if _num_conformers:
            dataset = PDBBind(
                cache_path=data_config.cache_path,
                split_path=data_config.split_train,
                keep_original=True,
                num_conformers=data_config.num_conformers,
                **common_args,
            )
        else:
            dataset = PDBBind(
                cache_path=data_config.cache_path,
                split_path=data_config.split_train,
                keep_original=True,
                **common_args,
            )

    elif mode == "validation":
        if _num_conformers:
            dataset = PDBBind(
                cache_path=data_config.cache_path,
                split_path=data_config.split_val,
                keep_original=True,
                num_conformers=data_config.num_conformers,
                **common_args,
            )
        else:
            dataset = PDBBind(
                cache_path=data_config.cache_path, split_path=data_config.split_val, keep_original=True, **common_args
            )

    elif mode == "test":
        if _num_conformers:
            dataset = PDBBind(
                cache_path=data_config.cache_path,
                split_path=data_config.split_test,
                keep_original=True,
                num_conformers=data_config.num_conformers,
                **common_args,
            )
        else:
            dataset = PDBBind(
                cache_path=data_config.cache_path, split_path=data_config.split_test, keep_original=True, **common_args
            )
    else:
        dataset = None

    return dataset
