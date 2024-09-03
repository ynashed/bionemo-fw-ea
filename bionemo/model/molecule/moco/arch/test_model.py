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
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch_scatter import scatter_mean

from bionemo.model.molecule.moco.arch.old_model import MoleculeDiT


ligand_pos = torch.rand((75, 3))
batch_ligand = torch.Tensor(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
    ]
).to(torch.int64)
ligand_feats = torch.Tensor(
    [
        2,
        4,
        2,
        4,
        2,
        4,
        4,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        5,
        1,
        3,
        1,
        1,
        1,
        2,
        4,
        2,
        4,
        2,
        4,
        4,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        5,
        1,
        3,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        12,
        2,
        5,
        2,
        3,
        5,
        1,
        5,
        2,
        4,
        2,
        4,
        2,
        4,
        4,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        5,
        1,
        3,
        1,
        1,
        1,
    ]
).to(torch.int64)
num_classes = 16

# Initialize the adjacency matrix with zeros
adj_matrix = torch.zeros((75, 75, 5), dtype=torch.int64)
no_bond = torch.zeros(5)
no_bond[0] = 1
# Using broadcasting to create the adjacency matrix
adj_matrix[batch_ligand.unsqueeze(1) == batch_ligand] = 1
for idx, i in enumerate(batch_ligand):
    for jdx, j in enumerate(batch_ligand):
        if idx == jdx:
            adj_matrix[idx][jdx] = no_bond
        elif i == j:
            adj_matrix[idx][jdx] = torch.nn.functional.one_hot(torch.randint(0, 5, (1,)), 5).squeeze(0)

atom_embedder = nn.Linear(num_classes, 64)
X = ligand_pos
X = X - scatter_mean(X, index=batch_ligand, dim=0, dim_size=X.shape[0])


# H = atom_embedder(F.one_hot(ligand_feats, num_classes).float())
H = F.one_hot(ligand_feats, num_classes).float()
A = adj_matrix
mask = batch_ligand.unsqueeze(1) == batch_ligand.unsqueeze(0)  # Shape: (75, 75)
E_idx = mask.nonzero(as_tuple=False).t()
self_loops = E_idx[0] != E_idx[1]
E_idx = E_idx[:, self_loops]
Z = atom_embedder(F.one_hot(ligand_feats, num_classes).float()).unsqueeze(1) * atom_embedder(
    F.one_hot(ligand_feats, num_classes).float()
).unsqueeze(0)
src, tgt = E_idx
E = A[src, tgt].float()

time = torch.tensor([0.2, 0.4, 0.6, 0.8])
model = MoleculeDiT()
print("Parameters", sum(p.numel() for p in model.parameters()))
model = model.cuda()
out = model(batch_ligand.cuda(), X.cuda(), H.cuda(), E_idx.cuda(), E.cuda(), time.cuda())


rotation = R.random().as_matrix()
rotation_tensor = torch.tensor(rotation, dtype=torch.float32)

X_rotated = torch.matmul(X, rotation_tensor.T)


out_rotated = model(batch_ligand.cuda(), X_rotated.cuda(), H.cuda(), E_idx.cuda(), E.cuda(), time.cuda())


torch.testing.assert_close(out_rotated['h_logits'], out['h_logits'], atol=1e-5, rtol=1e-5)

torch.testing.assert_close(out_rotated['edge_attr_logits'], out['edge_attr_logits'], atol=1e-5, rtol=1e-5)

torch.testing.assert_close(
    out_rotated["x_hat"], torch.matmul(out["x_hat"], rotation_tensor.T.cuda()), atol=1e-5, rtol=1e-5
)
