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

from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder
from bionemo.model.molecule.moco.metrics.metrics import BasicMolecularMetrics, get_molecules


class MoleculeEvaluationCallback(pl.Callback):
    def __init__(self, n_graphs=500, batch_size=100, timesteps=None):
        super().__init__()
        self.n_graphs = n_graphs
        self.batch_size = batch_size
        self.full_atom_decoder = full_atom_decoder
        self.timesteps = timesteps

    def evaluate_molecules(self, trainer, pl_module):
        mols = []
        with torch.no_grad():
            while len(mols) < self.n_graphs:
                current = min(self.n_graphs - len(mols), self.batch_size)
                generated = pl_module.sample(current, timesteps=self.timesteps)
                mols.extend(get_molecules(generated, {"atom_decoder": self.full_atom_decoder}))
        mol_metrics = BasicMolecularMetrics({"atom_decoder": self.full_atom_decoder}, device=pl_module.device)
        stab_dict, valid_dict, stat_dict, valid_smi, stable_mols, valid_mols = mol_metrics(mols)
        res = {**stab_dict, **valid_dict, **stat_dict}
        # import ipdb; ipdb.set_trace()
        if trainer.global_rank == 0:
            print(res)
        return res

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            res = self.evaluate_molecules(trainer, pl_module)
        except Exception as e:
            res = {'mol_stable': 0, 'atm_stable': 0, 'validity': 0, 'QED': -1, 'SA': -1, 'LogP': -1, 'Lipinski': -1}
            print(f"The sampling has been failed with the error: {e}")
        pl_module.log_dict(res, sync_dist=True)
