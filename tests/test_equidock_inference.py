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
import pathlib
import tempfile
from zipfile import ZipFile

import numpy as np
import pytest
import scipy.spatial as spa
import torch
from biopandas.pdb import PandasPdb
from hydra import compose, initialize

from bionemo.data.equidock.protein_utils import (
    get_coords,
)
from bionemo.model.protein.equidock.infer import EquiDockInference
from bionemo.model.protein.equidock.loss_metrics.eval import (
    Meter_Unbound_Bound,
)
from bionemo.model.protein.equidock.utils.train_utils import batchify_and_create_hetero_graphs_inference
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    check_model_exists,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


DATA_NAMES = ["dips", "db5"]
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, '../examples/protein/equidock/conf')

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
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer')
    cfg.data.data_name = data_name
    check_model_exists(cfg.model.restore_from_path)


@pytest.mark.needs_gpu
@pytest.mark.parametrize("data_name", DATA_NAMES)
def test_rmsds(data_name):
    method_name = 'equidock'

    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer')
    cfg.data.data_name = data_name
    # Set number of devices to 1 to be compatible with testing infra
    cfg.trainer.devices = 1
    cfg.exp_manager.exp_dir = None

    model = EquiDockInference(cfg=cfg)
    model.eval()

    # test data
    data_dir = os.path.join(
        THIS_FILE_DIR, 'equidock_test_data/test_sets_pdb/', f'{data_name}_test_random_transformed/random_transformed'
    )
    ground_truth_data_dir = os.path.join(
        THIS_FILE_DIR, 'equidock_test_data/test_sets_pdb/', f'{data_name}_test_random_transformed/complexes'
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

        pdb_files = [
            f
            for f in os.listdir(random_transformed_temp_dir)
            if os.path.isfile(os.path.join(random_transformed_temp_dir, f)) and f.endswith('.pdb')
        ]
        pdb_files.sort()
        cnt = 0

        for file in pdb_files:
            if cnt > 5:
                break

            if not file.endswith('_l_b.pdb'):
                continue

            ll = len('_l_b.pdb')
            ligand_filename = os.path.join(random_transformed_temp_dir, f'{file[:-ll]}_l_b.pdb')
            receptor_filename = os.path.join(ground_truth_temp_dir, f'{file[:-ll]}_r_b_COMPLEX.pdb')  # complexes
            out_filename = f'{file[:-ll]}_l_b_{method_name.upper()}.pdb'

            # Create ligand and receptor graphs and arrays
            (
                ligand_graph,
                receptor_graph,
                bound_ligand_repres_nodes_loc_clean_array,
                _,
            ) = model.model.create_ligand_receptor_graphs_arrays(ligand_filename, receptor_filename, cfg.data)

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
            assert (
                np.linalg.norm(new_residues - model_ligand_coors_deform_list[0].detach().cpu().numpy()) < 1e-1
            ), "Norm mismtach"

            ppdb_ligand = PandasPdb().read_pdb(ligand_filename)
            unbound_ligand_all_atoms_pre_pos = (
                ppdb_ligand.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
            )
            unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T + translation

            ppdb_ligand.df['ATOM'][
                ['x_coord', 'y_coord', 'z_coord']
            ] = unbound_ligand_new_pos  # unbound_ligand_new_pos
            unbound_ligand_save_filename = os.path.join(output_dir, out_filename)
            ppdb_ligand.to_pdb(path=unbound_ligand_save_filename, records=['ATOM'], gz=False)

            cnt += 1

        data_dir = output_dir
        pdb_files = [
            f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.pdb')
        ]
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

            assert ligand_model_coords.shape[0] == ligand_gt_coords.shape[0], "ligand shape mismatch"
            assert receptor_model_coords.shape[0] == receptor_gt_coords.shape[0], "receptor shape mismatch"

            ligand_receptor_distance = spa.distance.cdist(ligand_gt_coords, receptor_gt_coords)
            positive_tuple = np.where(ligand_receptor_distance < 8.0)
            active_ligand = positive_tuple[0]
            active_receptor = positive_tuple[1]
            ligand_model_pocket_coors = ligand_model_coords[active_ligand, :]
            receptor_model_pocket_coors = receptor_model_coords[active_receptor, :]
            ligand_gt_pocket_coors = ligand_gt_coords[active_ligand, :]
            receptor_gt_pocket_coors = receptor_gt_coords[active_receptor, :]

            crmsd = meter.update_rmsd(
                torch.Tensor(ligand_model_coords),
                torch.Tensor(receptor_model_coords),
                torch.Tensor(ligand_gt_coords),
                torch.Tensor(receptor_gt_coords),
            )

            irmsd = Irmsd_meter.update_rmsd(
                torch.Tensor(ligand_model_pocket_coors),
                torch.Tensor(receptor_model_pocket_coors),
                torch.Tensor(ligand_gt_pocket_coors),
                torch.Tensor(receptor_gt_pocket_coors),
            )

            all_crmsd.append(crmsd)
            all_irmsd.append(irmsd)

        expected_rmsd = np.load(os.path.join(THIS_FILE_DIR, f'equidock_test_data/expected_{data_name}_equidock.npz'))
        all_crmsd = np.array(all_crmsd)
        all_irmsd = np.array(all_irmsd)

        np.testing.assert_allclose(
            all_crmsd, expected_rmsd['crmsd'][:6], rtol=1e-3, atol=1e-2
        ), "Complex RMSD mismatch"
        np.testing.assert_allclose(
            all_irmsd, expected_rmsd['irmsd'][:6], rtol=1e-3, atol=1e-2
        ), "Interface RMSD mismatch"
