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
from torch_scatter import scatter_mean

from bionemo.model.molecule.moco.arch.dit import (
    MLP,
    XEGNN,
    BondRefine,
    DiTBlock,
    EdgeMixInLayer,
    EdgeMixOutLayer,
    SE3MixAttention,
)
from bionemo.model.molecule.moco.arch.scratch.mpnn import TimestepEmbedder
from bionemo.model.molecule.moco.models.utils import PredictionHead


class MoleculeDiT(nn.Module):
    def __init__(
        self,
        num_layers=8,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=16,
        edge_classes=5,
        num_heads=4,
    ):
        super(MoleculeDiT, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        #! TODO do we need coord prediction head which is mlp then 0 CoM?
        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.in_layer = EdgeMixInLayer(equivariant_node_feature_dim, invariant_node_feat_dim, invariant_edge_feat_dim)
        self.out_layer = EdgeMixOutLayer(
            equivariant_node_feature_dim, invariant_node_feat_dim, invariant_edge_feat_dim
        )
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.attention_mix_layers = nn.ModuleList()
        for i in range(num_layers):
            self.dit_layers.append(DiTBlock(invariant_node_feat_dim, num_heads, use_z=False))
            self.egnn_layers.append(XEGNN(invariant_node_feat_dim))
            self.attention_mix_layers.append(
                SE3MixAttention(equivariant_node_feature_dim, invariant_node_feat_dim, invariant_edge_feat_dim)
            )

    def forward(self, batch, X, H, E_idx, E, t):
        H = self.atom_embedder(H)
        E = self.edge_embedder(E)
        te = self.time_embedding(t, batch)

        X, H = self.in_layer(batch, X, H, E_idx, E)
        for layer_index in range(len(self.dit_layers)):
            H_dit = self.dit_layers[layer_index](batch, H, te)
            X_eg, H_eg = self.egnn_layers[layer_index](batch, X, H_dit, E_idx)
            # X = X_eg
            # H = H_eg
            X, H = self.attention_mix_layers[layer_index](batch, X_eg, H_eg, E_idx)
        X, H, edge_attr = self.out_layer(batch, X, H, E_idx)
        # import ipdb; ipdb.set_trace()
        edge_attr = self.bond_refine(batch, X, H, E_idx, edge_attr)
        x = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])
        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)
        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
            # "Z_hat": z_logits,
        }
        return out


if __name__ == "__main__":
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
    # out = model(batch_ligand, X, H, E_idx, E, time)
    loss = (out['x_hat'].sum()) ** 2
    loss.backward()
    import ipdb

    ipdb.set_trace()
    print("Success")
