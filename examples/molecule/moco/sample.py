# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch
from rdkit import Chem

from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder
from bionemo.model.molecule.moco.metrics.metrics import (
    BasicMolecularMetrics,
    get_molecules,
)
from bionemo.model.molecule.moco.models.module import Graph3DInterpolantModel
from bionemo.model.molecule.moco.models.utils_train import ExponentialMovingAverage as EMA


if __name__ == "__main__":
    # ckpt_path = "/workspace/bionemo/results/jodo/jodo_slurm/checkpoints/last.ckpt"
    # ckpt_path = "/workspace/bionemo/results/eqgatdiff/eqgat_slurm/checkpoints/last.ckpt"
    res_dir = "/workspace/bionemo/results/eqgatdiff/eqgat_big_self_cond/checkpoints"
    ckpt_path = f"{res_dir}/epoch=214-step=315199.ckpt"
    ema_weights = f"{res_dir}/ema_parameters_epoch_214.pt"
    n_graphs = 5000
    batch_size = 100
    save_n = 20
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)

    if ema_weights is not None:
        ema_weights = torch.load(ema_weights, map_location="cuda")
        ema = EMA(model.parameters(), decay=0.999)
        ema.load_state_dict(ema_weights["state_dict"], model.device)
        ema.copy_to(model.parameters())

    model.cuda()
    model.eval()
    mols = []
    with torch.no_grad():
        while len(mols) < n_graphs:
            current = min(n_graphs - len(mols), batch_size)
            generated = model.sample(current)
            mols.extend(get_molecules(generated, {"atom_decoder": full_atom_decoder}))

            mol_metrics = BasicMolecularMetrics({"atom_decoder": full_atom_decoder}, device=model.device)
            stab_dict, valid_dict, stat_dict, valid_smi, stable_mols, valid_mols = mol_metrics(mols)
            res = {**stab_dict, **valid_dict, **stat_dict}
            print(res)

    mol_metrics = BasicMolecularMetrics({"atom_decoder": full_atom_decoder}, device=model.device)
    stab_dict, valid_dict, stat_dict, valid_smi, stable_mols, valid_mols = mol_metrics(mols)
    res = {**stab_dict, **valid_dict, **stat_dict}
    print(res)

    idxs = set(np.random.randint(0, len(valid_mols), save_n))

    mols_to_save = [valid_mols[i].rdkit_mol for i in idxs]

    with Chem.SDWriter("_tmp/eqgat_sc.sdf") as w:
        for mol in mols_to_save:
            try:
                w.write(mol)
            except Exception:
                pass
