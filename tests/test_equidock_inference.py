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

import os
import tempfile

import pathlib
import pytest
from zipfile import ZipFile

import numpy as np
import torch

import scipy.spatial as spa
from biopandas.pdb import PandasPdb
from hydra import compose, initialize
from omegaconf import OmegaConf

from bionemo.data.equidock.protein_utils import preprocess_unbound_bound, protein_to_graph_unbound_bound, get_residues, get_rot_mat, get_coords
from bionemo.model.protein.equidock.utils.train_utils import batchify_and_create_hetero_graphs_inference

from bionemo.utils.tests import BioNemoSearchPathConfig, register_searchpath_config_plugin, update_relative_config_dir, check_model_exists

from bionemo.model.protein.equidock.loss_metrics.eval import Meter_Unbound_Bound
from bionemo.model.protein.equidock.infer import EquiDockInference

from bionemo.model.protein.equidock.loss_metrics.eval import rigid_transform_Kabsch_3D, rigid_transform_Kabsch_3D_torch

DATA_NAMES = ["dips", "db5"]
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, './conf')

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def extract_to_dir(zipfile, dir):
    with ZipFile(zipfile, 'r') as zipper:
        # extracting all the files
        zipper.extractall(dir)


def get_cfg(prepend_config_path, config_name, config_path='conf'):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    return cfg


@pytest.mark.parametrize("data_name", DATA_NAMES)
def test_model_exists(data_name):
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='equidock_test')
    cfg.data.data_name = data_name
    check_model_exists(cfg.model.restore_from_path)


@pytest.mark.needs_gpu
@pytest.mark.parametrize("data_name", DATA_NAMES)
def test_rmsds(data_name):
    method_name = 'equidock'

    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='equidock_test')
    cfg.data.data_name = data_name

    model = EquiDockInference(cfg=cfg)
    model.eval()

    # test data
    data_dir = os.path.join(
        THIS_FILE_DIR, 'equidock_test_data/test_sets_pdb/',
        f'{data_name}_test_random_transformed/random_transformed'
    )
    ground_truth_data_dir = os.path.join(
        THIS_FILE_DIR, 'equidock_test_data/test_sets_pdb/',
        f'{data_name}_test_random_transformed/complexes'
    )

    with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as ground_truth_temp_dir, tempfile.TemporaryDirectory() as random_transformed_temp_dir:

        # result directory
        output_dir = temp_dir

        # ground truth directory
        extract_to_dir(os.path.join(ground_truth_data_dir, 'ligands.zip'), ground_truth_temp_dir)
        extract_to_dir(os.path.join(ground_truth_data_dir, 'receptors.zip'), ground_truth_temp_dir)

        # random transformed directory
        extract_to_dir(os.path.join(data_dir, 'ligands.zip'), random_transformed_temp_dir)
        extract_to_dir(os.path.join(data_dir, 'receptors.zip'), random_transformed_temp_dir)

        pdb_files = [f for f in os.listdir(random_transformed_temp_dir) if os.path.isfile(
            os.path.join(random_transformed_temp_dir, f)) and f.endswith('.pdb')]
        pdb_files.sort()
        cnt = 0

        for file in pdb_files:
            if cnt > 5:
                break

            if not file.endswith('_l_b.pdb'):
                continue

            ll = len('_l_b.pdb')
            ligand_filename = os.path.join(random_transformed_temp_dir, f'{file[:-ll]}_l_b.pdb')
            receptor_filename = os.path.join(
                ground_truth_temp_dir, f'{file[:-ll]}_r_b_COMPLEX.pdb')  # complexes
            out_filename = f'{file[:-ll]}_l_b_{method_name.upper()}.pdb'

            ppdb_ligand = PandasPdb().read_pdb(ligand_filename)
            unbound_ligand_all_atoms_pre_pos = ppdb_ligand.df['ATOM'][[
                'x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)

            unbound_predic_ligand, unbound_predic_receptor, \
                bound_ligand_repres_nodes_loc_clean_array,\
                bound_receptor_repres_nodes_loc_clean_array = preprocess_unbound_bound(
                    get_residues(ligand_filename),
                    get_residues(receptor_filename),
                    graph_nodes=model.cfg.model.graph_nodes,
                    pos_cutoff=model.cfg.data.pocket_cutoff,
                    inference=True
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
                residue_loc_is_alphaC=model.cfg.model.graph_residue_loc_is_alphaC
            )

            if model.cfg.model.input_edge_feats_dim < 0:
                model.cfg.model.input_edge_feats_dim = ligand_graph.edata['he'].shape[1]

            ligand_graph.ndata['new_x'] = ligand_graph.ndata['x']

            assert np.linalg.norm(bound_ligand_repres_nodes_loc_clean_array -
                                  ligand_graph.ndata['x'].detach().cpu().numpy()) < 1e-1

            # Create a batch of a single DGL graph
            batch_hetero_graph = batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph)

            batch_hetero_graph = batch_hetero_graph.to(model.device)
            model_ligand_coors_deform_list, \
                model_keypts_ligand_list, model_keypts_receptor_list, \
                all_rotation_list, all_translation_list = model(batch_hetero_graph)

            rotation = all_rotation_list[0].detach().cpu().numpy()
            translation = all_translation_list[0].detach().cpu().numpy()

            new_residues = (rotation @ bound_ligand_repres_nodes_loc_clean_array.T).T + translation
            assert np.linalg.norm(
                new_residues - model_ligand_coors_deform_list[0].detach().cpu().numpy()) < 1e-1, f"Norm mismtach"

            unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T + translation

            euler_angles_finetune = torch.zeros([3], requires_grad=True)
            translation_finetune = torch.zeros([3], requires_grad=True)
            ligand_th = (get_rot_mat(euler_angles_finetune) @
                         torch.from_numpy(unbound_ligand_new_pos).T).T + translation_finetune

            ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']
                                   ] = ligand_th.detach().numpy()  # unbound_ligand_new_pos
            unbound_ligand_save_filename = os.path.join(output_dir, out_filename)
            ppdb_ligand.to_pdb(path=unbound_ligand_save_filename, records=['ATOM'], gz=False)

            cnt += 1

        data_dir = output_dir
        pdb_files = [f for f in os.listdir(data_dir) if os.path.isfile(
            os.path.join(data_dir, f)) and f.endswith('.pdb')]
        pdb_files.sort()

        meter = Meter_Unbound_Bound()
        Irmsd_meter = Meter_Unbound_Bound()

        all_crmsd = []
        all_irmsd = []

        for file in pdb_files:
            if cnt < 0:
                break

            if not file.endswith(f'_l_b_{method_name.upper()}.pdb'):
                continue
            cnt -= 1
            ll = len(f'_l_b_{method_name.upper()}.pdb')
            ligand_model_file = os.path.join(data_dir, f'{file[:-ll]}_l_b_{method_name.upper()}.pdb')
            ligand_gt_file = os.path.join(ground_truth_temp_dir, f'{file[:-ll]}_l_b_COMPLEX.pdb')
            receptor_model_file = os.path.join(ground_truth_temp_dir, f'{file[:-ll]}_r_b_COMPLEX.pdb')
            receptor_gt_file = os.path.join(ground_truth_temp_dir, f'{file[:-ll]}_r_b_COMPLEX.pdb')

            ligand_model_coords = get_coords(ligand_model_file, all_atoms=False)
            receptor_model_coords = get_coords(receptor_model_file, all_atoms=False)

            ligand_gt_coords = get_coords(ligand_gt_file, all_atoms=False)
            receptor_gt_coords = get_coords(receptor_gt_file, all_atoms=False)

            assert ligand_model_coords.shape[0] == ligand_gt_coords.shape[0], f"ligand shape mismatch"
            assert receptor_model_coords.shape[0] == receptor_gt_coords.shape[0], f"receptor shape mismatch"

            ligand_receptor_distance = spa.distance.cdist(ligand_gt_coords, receptor_gt_coords)
            positive_tuple = np.where(ligand_receptor_distance < 8.)
            active_ligand = positive_tuple[0]
            active_receptor = positive_tuple[1]
            ligand_model_pocket_coors = ligand_model_coords[active_ligand, :]
            receptor_model_pocket_coors = receptor_model_coords[active_receptor, :]
            ligand_gt_pocket_coors = ligand_gt_coords[active_ligand, :]
            receptor_gt_pocket_coors = receptor_gt_coords[active_receptor, :]

            crmsd = meter.update_rmsd(torch.Tensor(ligand_model_coords), torch.Tensor(receptor_model_coords),
                                      torch.Tensor(ligand_gt_coords), torch.Tensor(receptor_gt_coords))

            irmsd = Irmsd_meter.update_rmsd(torch.Tensor(ligand_model_pocket_coors), torch.Tensor(receptor_model_pocket_coors),
                                            torch.Tensor(ligand_gt_pocket_coors), torch.Tensor(receptor_gt_pocket_coors))

            all_crmsd.append(crmsd)
            all_irmsd.append(irmsd)

        expected_rmsd = np.load(
            os.path.join(THIS_FILE_DIR, f'equidock_test_data/expected_{data_name}_equidock.npz')
        )
        all_crmsd = np.array(all_crmsd)
        all_irmsd = np.array(all_irmsd)

        np.testing.assert_allclose(
            all_crmsd, expected_rmsd['crmsd'][:6], rtol=1e-3, atol=1e-2), f"Complex RMSD mismatch"
        np.testing.assert_allclose(
            all_irmsd, expected_rmsd['irmsd'][:6], rtol=1e-3, atol=1e-2), f"Interface RMSD mismatch"


@pytest.mark.needs_gpu
@pytest.mark.parametrize("data_type", [torch.float32, torch.float64])
def test_Kabsch_algorithm(data_type):
    for _ in range(10):
        # Random points
        A = np.random.randn(3, 10)
        B = np.random.randn(3, 10)

        # Get transformations from both functions
        R_numpy, t_numpy = rigid_transform_Kabsch_3D(A, B)
        R_torch, t_torch = rigid_transform_Kabsch_3D_torch(torch.from_numpy(A).to(
            'cuda').to(data_type), torch.from_numpy(B).to('cuda').to(data_type))

        # Convert the torch tensors to numpy arrays for easy comparison
        R_torch = R_torch.detach().cpu().numpy()
        t_torch = t_torch.detach().cpu().numpy()

        # Assert if matrices and vectors are close
        assert np.allclose(R_numpy, R_torch, atol=1e-4 if data_type ==
                           torch.float32 else 1e-6), f"Rotation matrices differ: \n{R_numpy}\n{R_torch}"
        assert np.allclose(t_numpy, t_torch, atol=1e-4 if data_type ==
                           torch.float32 else 1e-6), f"Translation vectors differ: \n{t_numpy}\n{t_torch}"
