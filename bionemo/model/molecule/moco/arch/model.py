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
from bionemo.model.molecule.moco.arch.dit_eqf import (
    MultidimEquiUpdate,
    MultiEdgeMixInLayer,
    MultiEdgeMixOutLayer,
    MultiXEGNN,
)
from bionemo.model.molecule.moco.arch.dit_mpnn import DiTeMPNN
from bionemo.model.molecule.moco.arch.dite import DiTeBlock, MultiXEGNNE, MultiXEGNNET
from bionemo.model.molecule.moco.arch.scratch.mpnn import TimestepEmbedder
from bionemo.model.molecule.moco.models.utils import PredictionHead


def coord2dist(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    return radial


class MoleculeDiTeMPNN(nn.Module):
    def __init__(
        self,
        num_layers=8,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=16,
        edge_classes=5,
        num_heads=4,
        n_vector_features=128,
    ):
        super(MoleculeDiTeMPNN, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)

        #! TODO do we need coord prediction head which is mlp then 0 CoM?
        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.attention_mix_layers = nn.ModuleList()
        for i in range(num_layers):
            self.dit_layers.append(DiTeMPNN(invariant_node_feat_dim, num_heads, use_z=False))
            self.egnn_layers.append(MultiXEGNNET(invariant_node_feat_dim))
        # self.h_feat_refine = DiTBlock(invariant_node_feat_dim, num_heads, use_z=False)
        # self.edge_feat_refine = DiTBlock(invariant_node_feat_dim, num_heads, use_z=False)

    def forward(self, batch, X, H, E_idx, E, t):
        torch.max(batch) + 1
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K

        H = self.atom_embedder(H)
        E = self.edge_embedder(E)  # should be + n_vector_features not + 1
        te = self.time_embedding(t)
        te_h = te[batch]
        # edge_batch, counts = torch.unique(batch, return_counts=True)
        # edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  #
        edge_batch = batch[E_idx[0]]
        te_e = te[edge_batch]
        edge_attr = E
        for layer_index in range(len(self.dit_layers)):
            distances = coord2dist(pos, E_idx).squeeze(1)  # E x K
            H, edge_attr = self.dit_layers[layer_index](batch, H, te_h, edge_attr, E_idx, te_e, distances)
            pos = self.egnn_layers[layer_index](batch, pos, H, E_idx, edge_attr, te_e)  #! TODO at time here

        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]
        # import ipdb; ipdb.set_trace()
        # H = self.h_feat_refine(batch, H, te_h)
        # edge_attr = self.edge_feat_refine(edge_batch, edge_attr, te_e) #! cannot do attn with edges due to memory

        edge_attr = self.bond_refine(batch, X, H, E_idx, edge_attr)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)
        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
            # "Z_hat": z_logits,
        }
        return out


class MoleculeDiTe2(nn.Module):
    def __init__(
        self,
        num_layers=8,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=16,
        edge_classes=5,
        num_heads=4,
        n_vector_features=128,
    ):
        super(MoleculeDiTe2, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)

        #! TODO do we need coord prediction head which is mlp then 0 CoM?
        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.attention_mix_layers = nn.ModuleList()
        for i in range(num_layers):
            self.dit_layers.append(DiTeBlock(invariant_node_feat_dim, num_heads, use_z=False))
            self.egnn_layers.append(MultiXEGNNET(invariant_node_feat_dim))
        self.h_feat_refine = DiTBlock(invariant_node_feat_dim, num_heads, use_z=False)
        # self.edge_feat_refine = DiTBlock(invariant_node_feat_dim, num_heads, use_z=False)

    def forward(self, batch, X, H, E_idx, E, t):
        torch.max(batch) + 1
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K

        H = self.atom_embedder(H)
        E = self.edge_embedder(E)  # should be + n_vector_features not + 1
        te = self.time_embedding(t)
        te_h = te[batch]
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  #
        te_e = te[edge_batch]
        edge_attr = E
        for layer_index in range(len(self.dit_layers)):
            distances = coord2dist(pos, E_idx).squeeze(1)  # E x K
            H, edge_attr = self.dit_layers[layer_index](batch, H, te_h, edge_attr, E_idx, te_e, distances, edge_batch)
            pos = self.egnn_layers[layer_index](batch, pos, H, E_idx, edge_attr, te_e)  #! TODO at time here

        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]
        # import ipdb; ipdb.set_trace()
        H = self.h_feat_refine(batch, H, te_h)
        # edge_attr = self.edge_feat_refine(edge_batch, edge_attr, te_e) #! cannot do attn with edges due to memory

        edge_attr = self.bond_refine(batch, X, H, E_idx, edge_attr)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)
        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
            # "Z_hat": z_logits,
        }
        return out


class MoleculeDiTe(nn.Module):
    def __init__(
        self,
        num_layers=8,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=16,
        edge_classes=5,
        num_heads=4,
        n_vector_features=128,
    ):
        super(MoleculeDiTe, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)

        #! TODO do we need coord prediction head which is mlp then 0 CoM?
        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.attention_mix_layers = nn.ModuleList()
        for i in range(num_layers):
            self.dit_layers.append(DiTeBlock(invariant_node_feat_dim, num_heads, use_z=False))
            self.egnn_layers.append(MultiXEGNNE(invariant_node_feat_dim))

    def forward(self, batch, X, H, E_idx, E, t):
        torch.max(batch) + 1
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K

        H = self.atom_embedder(H)
        E = self.edge_embedder(E)  # should be + n_vector_features not + 1
        te = self.time_embedding(t)
        te_h = te[batch]
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  #
        te_e = te[edge_batch]
        edge_attr = E
        for layer_index in range(len(self.dit_layers)):
            distances = coord2dist(pos, E_idx).squeeze(1)  # E x K
            H, edge_attr = self.dit_layers[layer_index](batch, H, te_h, edge_attr, E_idx, te_e, distances, edge_batch)
            pos = self.egnn_layers[layer_index](batch, pos, H, E_idx, edge_attr)  #! TODO at time here

        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]
        # import ipdb; ipdb.set_trace()
        edge_attr = self.bond_refine(batch, X, H, E_idx, edge_attr)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)
        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
            # "Z_hat": z_logits,
        }
        return out


class MoleculeVecDiT(nn.Module):
    def __init__(
        self,
        num_layers=8,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=16,
        edge_classes=5,
        num_heads=4,
        n_vector_features=128,
    ):
        super(MoleculeVecDiT, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes + n_vector_features, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)

        #! TODO do we need coord prediction head which is mlp then 0 CoM?
        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)

        self.mix_layer = MultidimEquiUpdate(
            invariant_node_feat_dim,
            invariant_edge_feat_dim,
            n_vector_features,
            invariant_node_feat_dim,
            n_vector_features=n_vector_features,
        )

        self.in_layer = MultiEdgeMixInLayer(
            equivariant_node_feature_dim, invariant_node_feat_dim, invariant_edge_feat_dim
        )
        self.out_layer = MultiEdgeMixOutLayer(
            equivariant_node_feature_dim, invariant_node_feat_dim, invariant_edge_feat_dim
        )
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.attention_mix_layers = nn.ModuleList()
        for i in range(num_layers):
            self.dit_layers.append(DiTBlock(invariant_node_feat_dim, num_heads, use_z=False))
            self.egnn_layers.append(MultiXEGNN(invariant_node_feat_dim))
            # self.attention_mix_layers.append(
            #     SE3MixAttention(equivariant_node_feature_dim, invariant_node_feat_dim, invariant_edge_feat_dim)
            # )

    def forward(self, batch, X, H, E_idx, E, t):
        torch.max(batch) + 1
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K
        distances = coord2dist(pos, E_idx)
        # import ipdb; ipdb.set_trace()
        distances = distances.squeeze(1)  # E x K
        H = self.atom_embedder(H)
        E = self.edge_embedder(torch.cat([E, distances], dim=-1))  # should be + n_vector_features not + 1
        te = self.time_embedding(t, batch)
        #! Model missing proper mixing, equivariant features and distance and H interaction || Z tensor could fix this along with proepr mixing
        X, H = self.in_layer(batch, pos, H, E_idx, E)
        for layer_index in range(len(self.dit_layers)):
            H_dit = self.dit_layers[layer_index](batch, H, te)
            X, H = self.egnn_layers[layer_index](batch, X, H_dit, E_idx)
            # pos, H = self.attention_mix_layers[layer_index](batch, X_eg, H_eg, E_idx)

        X, H, edge_attr = self.out_layer(batch, X, H, E_idx)
        X = self.coord_pred(X).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]
        # import ipdb; ipdb.set_trace()
        edge_attr = self.bond_refine(batch, X, H, E_idx, edge_attr)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)
        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
            # "Z_hat": z_logits,
        }
        return out


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
        #! Model missing proper mixing, equivariant features and distance and H interaction || Z tensor could fix this along with proepr mixing
        X, H = self.in_layer(batch, X, H, E_idx, E)
        for layer_index in range(len(self.dit_layers)):
            rel_vec = X.unsqueeze(1) - X.unsqueeze(0)  # [b, a, a, 3]
            rel_dist = (rel_vec**2).sum(dim=-1, keepdim=True)
            square_batch = (batch.unsqueeze(1) == batch.unsqueeze(0)).int()
            torch.ones_like(square_batch) - torch.eye(
                square_batch.size(0), dtype=square_batch.dtype, device=square_batch.device
            )
            rel_dist = rel_dist * square_batch.unsqueeze(-1)

            import ipdb

            ipdb.set_trace()
            H_dit = self.dit_layers[layer_index](batch, H, te)
            X_eg, H_eg = self.egnn_layers[layer_index](batch, X, H_dit, E_idx)
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
