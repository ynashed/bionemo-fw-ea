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
import time
from collections import defaultdict

import torch

from bionemo.model.molecule.moco.data.data_utils import full_atom_encoder
from bionemo.model.molecule.moco.data.molecule_datamodule import MoleculeDataModule
from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder
from bionemo.model.molecule.moco.metrics.molecule import get_molecules
from bionemo.model.molecule.moco.metrics.molecule_metrics_3d import Molecule3DMetrics
from bionemo.model.molecule.moco.metrics.molecule_stability_2d import Molecule2DStability
from bionemo.model.molecule.moco.models.module import Graph3DInterpolantModel


if __name__ == "__main__":
    best_ckpt_path = "/workspace/bionemo/bionemo/model/molecule/moco/ckpt/julian_best_mol_stab.ckpt"
    ckpt_path = "/workspace/bionemo/bionemo/model/molecule/moco/models/results/eqgatdiff/EQGAT_FW_TEST_6_28_charges_live_interpolant/checkpoints/last.ckpt"
    save_path = '/workspace/bionemo/bionemo/model/molecule/moco/models/results/eqgatdiff/julian/'  #! set up saving for infernece and forward

    state_dict = torch.load(best_ckpt_path)["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
    model.dynamics.load_state_dict(state_dict)
    data_config = {
        "dataset_root": "/workspace/bionemo/data/pyg_geom_drug",
        "processed_folder": "processed",
        "batch_size": 100,  # 200
        "inference_batch_size": 100,  # 200
        "removed_h": False,
        "data_loader_type": "midi",
    }
    datamodule = MoleculeDataModule(**data_config)
    statistics = datamodule.statistics
    dataset_info = {
        "atom_decoder": full_atom_decoder,
        "atom_encoder": full_atom_encoder,
        "statistics": statistics,
    }
    print("Double Check reloading from correct dictionary for resume right now loading from old but building _new")
    # assert(1 == 0)
    model.cuda()
    model.eval()
    METS = {}
    with open(save_path + "eqgat_lengths_metrics.pkl", 'rb') as f:
        METS = pickle.load(f)
    OUTPUT = defaultdict(list)
    with open(save_path + "eqgat_lengths_new.pkl", 'rb') as f:
        OUTPUT = pickle.load(f)
    amount = 100
    # sizes = list(range(30,151))

    sizes = list(range(30, 126))
    MODEL_OUT = {}
    with open(save_path + "eqgat_lengths_model_out_new.pkl", 'rb') as f:
        MODEL_OUT = pickle.load(f)
    for size in sizes:
        if size in METS and size in OUTPUT:
            print(f"size {size} already exists ... skipping to next")
            continue
        mol_2d_stability = Molecule2DStability({"atom_decoder": full_atom_decoder}, device=model.device)
        mol_3d_metrics = Molecule3DMetrics(dataset_info, device=model.device, compute_dihedrals=True)

        print(f"Generating {amount} molecules with {size} number of atoms")
        if size < 54:  #! GPU memory error
            batch_split = 1
            amounts = [100]
        elif size < 93:
            batch_split = 2
            amounts = [int(amount / batch_split)] * batch_split
        else:
            batch_split = 3
            amounts = [34, 33, 33]

        # amounts = [int(amount / batch_split)] * batch_split
        print(amounts)
        start = time.time()
        outs = []
        for amo in amounts:
            num_atoms = torch.tensor([size] * amo)
            out = model.sample(amo, num_atoms=num_atoms)
            outs.append(out)
        end = time.time()
        mols = []
        for out in outs:
            mol = get_molecules(out, dataset_info)
            mols.extend(mol)
        stability_res, valid_smiles, valid_molecules, stable_molecules, info_2d = mol_2d_stability(mols)
        mol_3d_res = mol_3d_metrics(mols)
        stability_res['time'] = end - start
        stability_res['batch_size'] = (amount / batch_split, batch_split)
        print(stability_res, mol_3d_res)
        METS[size] = {"2d_metrics": stability_res, "3d_metrics": mol_3d_res}
        with open(save_path + "eqgat_lengths_metrics.pkl", 'wb') as f:
            pickle.dump(METS, f)
        OUTPUT[size] = []
        OUTPUT[size].append(
            {"2d_metrics": stability_res, "3d_metrics": mol_3d_res, "mols": [x.rdkit_mol for x in mols]}
        )
        with open(save_path + "eqgat_lengths_new.pkl", 'wb') as f:
            pickle.dump(OUTPUT, f)
        MODEL_OUT[size] = mols
        with open(save_path + "eqgat_lengths_model_out_new.pkl", 'wb') as f:
            pickle.dump(MODEL_OUT, f)
