# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from lightning import pytorch as pl

from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder, full_atom_encoder
from bionemo.model.molecule.moco.metrics.molecule import get_molecules
from bionemo.model.molecule.moco.metrics.molecule_metrics_2d import Molecule2DMetrics
from bionemo.model.molecule.moco.metrics.molecule_metrics_3d import Molecule3DMetrics
from bionemo.model.molecule.moco.metrics.molecule_novelty_similarity import MoleculeTrainDataMetrics
from bionemo.model.molecule.moco.metrics.molecule_stability_2d import Molecule2DStability


class MoleculeEvaluationCallback(pl.Callback):
    """
    Callback for evaluating generated molecules on various metrics.

    Args:
        n_graphs (int): Number of molecules to generate for evaluation.
        batch_size (int): Batch size for molecule generation.
        timesteps (Optional[int]): Number of timesteps for sampling.
        train_smiles (Optional[List[str]]): Training dataset SMILES strings for calculating similarity and novelty.
        statistics (Optional[Dict]): Precomputed dataset statistics.
        compute_2D_metrics (bool): Whether to compute 2D metrics.
        compute_3D_metrics (bool): Whether to compute 3D metrics.
        compute_dihedrals (bool): Whether to compute dihedral angles in 3D metrics.
        compute_train_data_metrics (bool): Whether to compute train data metrics (similarity and novelty).
    """

    def __init__(
        self,
        n_graphs=500,
        batch_size=100,
        timesteps=None,
        train_smiles=None,
        statistics=None,
        compute_2D_metrics=True,
        compute_3D_metrics=True,
        compute_dihedrals=False,
        compute_train_data_metrics=True,
    ):
        super().__init__()
        self.n_graphs = n_graphs
        self.batch_size = batch_size
        self.full_atom_decoder = full_atom_decoder
        self.timesteps = timesteps
        self.train_smiles = train_smiles
        self.dataset_info = {
            "atom_decoder": full_atom_decoder,
            "atom_encoder": full_atom_encoder,
            "statistics": statistics,
        }
        self.compute_2D_metrics = compute_2D_metrics
        self.compute_3D_metrics = compute_3D_metrics
        self.compute_dihedrals = compute_dihedrals
        self.compute_train_data_metrics = compute_train_data_metrics

    def gather_default_values(self):
        """
        Gather default values for metrics when evaluation fails.

        Returns:
            Dict: Default values for each metric.
        """
        defaults = {}
        defaults.update(Molecule2DStability.default_values())
        if self.compute_2D_metrics:
            defaults.update(Molecule2DMetrics.default_values())
        if self.compute_3D_metrics:
            defaults.update(Molecule3DMetrics.default_values())
        if self.compute_train_data_metrics:
            defaults.update(MoleculeTrainDataMetrics.default_values())
        return defaults

    def evaluate_molecules(self, pl_module, trainer=None):
        """
        Evaluate generated molecules on specified metrics.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: PyTorch Lightning module instance.

        Returns:
            Dict: Results of the evaluation.
        """
        mols = []
        with torch.no_grad():
            while len(mols) < self.n_graphs:
                current = min(self.n_graphs - len(mols), self.batch_size)
                generated = pl_module.sample(current, timesteps=self.timesteps)
                mols.extend(get_molecules(generated, {"atom_decoder": self.full_atom_decoder}))

        results = {}

        # Evaluate 2D stability
        mol_2d_stability = Molecule2DStability(self.dataset_info, device=pl_module.device)
        stability_res, valid_smiles, valid_molecules, stable_molecules, info_2d = mol_2d_stability(mols)
        results.update(stability_res)

        if self.compute_2D_metrics:
            # Evaluate 2D metrics
            mol_2d_metrics = Molecule2DMetrics(self.dataset_info, device=pl_module.device)
            statistics_res = mol_2d_metrics.evaluate(valid_smiles)
            results.update(statistics_res)

        if self.compute_3D_metrics:
            # Evaluate 3D metrics
            mol_3d_metrics = Molecule3DMetrics(
                self.dataset_info, device=pl_module.device, compute_dihedrals=self.compute_dihedrals
            )
            mol_3d_res = mol_3d_metrics(mols)
            results.update(mol_3d_res)

        if self.compute_train_data_metrics:
            # Evaluate train data metrics
            train_data_metrics = MoleculeTrainDataMetrics(self.train_smiles, device=pl_module.device)
            train_data_res = train_data_metrics(valid_smiles)
            results.update(train_data_res)

        results['store_valid_molecules'] = valid_molecules
        results['store_stable_molecules'] = stable_molecules
        results['store_info_2d'] = info_2d
        if trainer is None or trainer.global_rank == 0:
            for key, value in results.items():
                if "store" not in key:
                    print(key, value)
        return results

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the validation epoch to evaluate molecules and log metrics.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: PyTorch Lightning module instance.
        """
        try:
            results = self.evaluate_molecules(pl_module, trainer=trainer)
        except Exception as e:
            results = self.gather_default_values()
            print(f"The sampling has failed with the error: {e}")
        pl_module.log_dict(results, sync_dist=True)
