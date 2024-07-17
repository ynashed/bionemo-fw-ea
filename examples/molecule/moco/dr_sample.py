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

import torch

from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder
from bionemo.model.molecule.moco.metrics.metrics import (
    BasicMolecularMetrics,
    get_molecules,
)
from bionemo.model.molecule.moco.models.module import Graph3DInterpolantModel


if __name__ == "__main__":
    # ckpt_path = "/workspace/bionemo/results/jodo/JODO_CHARGES/checkpoints/last-v1.ckpt"
    # model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
    # model.cuda()
    # model.eval()
    # with torch.no_grad():
    #     mols = model.sample(100)
    # mol_metrics = BasicMolecularMetrics({"atom_decoder": full_atom_decoder}, device=model.device)
    # mols = get_molecules(mols, {"atom_decoder": full_atom_decoder})
    # stab_dict, valid_dict, stat_dict, valid_smi, stable_mols, valid_mols = mol_metrics(mols)
    # res = {**stab_dict, **valid_dict, **stat_dict}
    # print(res)

    best_ckpt_path = "/workspace/bionemo/bionemo/model/molecule/moco/ckpt/julian_best_mol_stab.ckpt"
    ckpt_path = "/workspace/bionemo/bionemo/model/molecule/moco/models/results/eqgatdiff/EQGAT_FW_TEST_6_28_charges_live_interpolant/checkpoints/last.ckpt"
    save_path = '/workspace/bionemo/bionemo/model/molecule/moco/models/results/eqgatdiff/julian/'  #! set up saving for infernece and forward
    state_dict = torch.load(best_ckpt_path)["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    n_graphs = 1000
    batch_size = 100
    save_n = 20
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
    model.dynamics.load_state_dict(state_dict)

    model.cuda()
    model.eval()
    with torch.no_grad():
        mols, mol_traj = model.sample(100, timesteps=400, save_all=True)
    mol_metrics = BasicMolecularMetrics({"atom_decoder": full_atom_decoder}, device=model.device)
    mols = get_molecules(mols, {"atom_decoder": full_atom_decoder})
    stab_dict, valid_dict, stat_dict, valid_smi, stable_mols, valid_mols = mol_metrics(mols)
    res = {**stab_dict, **valid_dict, **stat_dict}
    print(res)
    safe_mols = [x.rdkit_mol for x in mols]
    with open(save_path + "sample_400_new.pkl", 'wb') as f:
        pickle.dump((safe_mols, res), f)
    with open(save_path + "sample_400_new_traj.pkl", 'wb') as f:
        pickle.dump((mol_traj, res), f)
    import ipdb

    ipdb.set_trace()
    test = 1
    # data_params = {
    #     "dataset_root": "/workspace/bionemo/data/pyg_geom_drug",
    #     "processed_folder": "processed",
    #     "batch_size": 150,
    #     "inference_batch_size": 150,
    #     "removed_h": False,
    #     "data_loader_type": "midi",
    #  }

    # datamodule = MoleculeDataModule(**data_params)
    # # import ipdb; ipdb.set_trace()
    # train_loader = datamodule.train_dataloader()
    # for batch in train_loader:
    #     batch = batch.to(model.device)
    #     batch.h = batch.x
    #     batch.x = batch.pos
    #     batch.pos = None
    #     time = model.sample_time(batch)
    #     out, batch, time = model(batch, time)
    #     result = {
    #         "out": out,
    #         "batch": batch,
    #         "time": time
    #     }
    #     with open(save_path + "train_batch.pkl", 'wb') as f:
    #         pickle.dump(result, f)
    #     break
    # import ipdb; ipdb.set_trace()
    # test = 1
