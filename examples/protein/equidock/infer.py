#!/bin/bash

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

"""
Entry point to EquiDock.

modify parameters from conf/*.cfg
"""
import os
import pathlib
import tempfile
import time

import numpy as np
import torch
from biopandas.pdb import PandasPdb
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import bionemo
from bionemo.data.equidock.protein_utils import (
    extract_to_dir,
    get_residues,
    get_rot_mat,
    preprocess_unbound_bound,
    protein_to_graph_unbound_bound,
)
from bionemo.model.protein.equidock.infer import EquiDockInference
from bionemo.model.protein.equidock.utils.train_utils import batchify_and_create_hetero_graphs_inference


os.environ['DGLBACKEND'] = 'pytorch'
torch.set_float32_matmul_precision("high")
BIONEMO_ROOT = pathlib.Path(bionemo.__file__).parent.parent.as_posix()


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    logging.info(f"\nTrain dataset name {cfg.data.data_name}")

    seed_everything(cfg.seed)

    #######################################
    # set data IO dirs
    #######################################
    output_dir = cfg.exp_manager.exp_dir
    #######################################
    # load model and chpt from cfg
    # model = EquiDockInference(cfg)
    #######################################
    logging.info("\n\n************** Loading EquiDockInference ***********")
    model = EquiDockInference(cfg=cfg)
    model.eval()
    data_dir = cfg.data.data_dir

    with tempfile.TemporaryDirectory() as temp_dir:
        # random transformed zip pdb directory
        extract_to_dir(os.path.join(data_dir, 'ligands.zip'), temp_dir)
        extract_to_dir(os.path.join(data_dir, 'receptors.zip'), temp_dir)

        pdb_files = [
            f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f)) and f.endswith('.pdb')
        ]
        pdb_files.sort()
        time_list = []

        for file in pdb_files:
            start = time.time()
            if not file.endswith('_l_b.pdb'):
                continue

            ll = len('_l_b.pdb')
            ligand_filename = os.path.join(temp_dir, f'{file[:-ll]}_l_b.pdb')
            receptor_filename = os.path.join(temp_dir, f'{file[:-ll]}_r_b.pdb')
            out_filename = f'{file[:-ll]}_l_b_COMPLEX.pdb'
            logging.info(f"Processing {out_filename} ...")

            ppdb_ligand = PandasPdb().read_pdb(ligand_filename)
            unbound_ligand_all_atoms_pre_pos = (
                ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
            )

            (
                unbound_predic_ligand,
                unbound_predic_receptor,
                bound_ligand_repres_nodes_loc_clean_array,
                bound_receptor_repres_nodes_loc_clean_array,
            ) = preprocess_unbound_bound(
                get_residues(ligand_filename),
                get_residues(receptor_filename),
                graph_nodes=model.cfg.model.graph_nodes,
                pos_cutoff=model.cfg.data.pocket_cutoff,
                inference=True,
            )

            ligand_graph, receptor_graph = protein_to_graph_unbound_bound(
                unbound_predic_ligand,
                unbound_predic_receptor,
                bound_ligand_repres_nodes_loc_clean_array,
                bound_receptor_repres_nodes_loc_clean_array,
                graph_nodes=model.cfg.model.graph_nodes,
                cutoff=model.cfg.data.graph_cutoff,
                max_neighbor=model.cfg.data.graph_max_neighbor,
                one_hot=False,
                residue_loc_is_alphaC=model.cfg.model.graph_residue_loc_is_alphaC,
            )

            if model.cfg.model.input_edge_feats_dim < 0:
                model.cfg.model.input_edge_feats_dim = ligand_graph.edata['he'].shape[1]

            ligand_graph.ndata['new_x'] = ligand_graph.ndata['x']

            assert (
                np.linalg.norm(
                    bound_ligand_repres_nodes_loc_clean_array - ligand_graph.ndata['x'].detach().cpu().numpy()
                )
                < 1e-1
            )

            # Create a batch of a single DGL graph
            batch_hetero_graph = batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph)

            batch_hetero_graph = batch_hetero_graph.to(model.device)
            (
                model_ligand_coors_deform_list,
                model_keypts_ligand_list,
                model_keypts_receptor_list,
                all_rotation_list,
                all_translation_list,
            ) = model(batch_hetero_graph)

            rotation = all_rotation_list[0].detach().cpu().numpy()
            translation = all_translation_list[0].detach().cpu().numpy()

            new_residues = (rotation @ bound_ligand_repres_nodes_loc_clean_array.T).T + translation
            assert np.linalg.norm(new_residues - model_ligand_coors_deform_list[0].detach().cpu().numpy()) < 1e-1

            unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T + translation

            euler_angles_finetune = torch.zeros([3], requires_grad=True)
            translation_finetune = torch.zeros([3], requires_grad=True)
            ligand_th = (
                get_rot_mat(euler_angles_finetune) @ torch.from_numpy(unbound_ligand_new_pos).T
            ).T + translation_finetune

            ppdb_ligand.df['ATOM'][
                ['x_coord', 'y_coord', 'z_coord']
            ] = ligand_th.detach().numpy()  # unbound_ligand_new_pos
            unbound_ligand_save_filename = os.path.join(output_dir, out_filename)
            ppdb_ligand.to_pdb(path=unbound_ligand_save_filename, records=['ATOM'], gz=False)

            time_list.append((time.time() - start))

        time_array = np.array(time_list)
        logging.info(f"Mean runtime: {np.mean(time_array)}, std runtime: {np.std(time_array)}")
        logging.info(f"\nResults are written into {output_dir}")


if __name__ == "__main__":
    main()
