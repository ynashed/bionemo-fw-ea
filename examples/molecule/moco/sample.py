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

from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder
from bionemo.model.molecule.moco.metrics.metrics import (
    BasicMolecularMetrics,
    get_molecules,
)
from bionemo.model.molecule.moco.models.module import Graph3DInterpolantModel


if __name__ == "__main__":
    ckpt_path = "/workspace/bionemo/results/jodo/JODO_CHARGES/checkpoints/last-v1.ckpt"
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
    model.cuda()
    model.eval()
    with torch.no_grad():
        mols = model.sample(100)
    mol_metrics = BasicMolecularMetrics({"atom_decoder": full_atom_decoder}, device=model.device)
    mols = get_molecules(mols, {"atom_decoder": full_atom_decoder})
    stab_dict, valid_dict, stat_dict, valid_smi, stable_mols, valid_mols = mol_metrics(mols)
    res = {**stab_dict, **valid_dict, **stat_dict}
    print(res)
