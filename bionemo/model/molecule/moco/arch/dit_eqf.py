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
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
from torch_scatter import scatter, scatter_mean

from bionemo.model.molecule.moco.arch.dit import MLP, modulate


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "gelu_tanh": nn.GELU(approximate='tanh'),
    "sigmoid": nn.Sigmoid(),
}


class MultidimE3Norm(nn.Module):
    def __init__(self, n_vector_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((1, 1, n_vector_features)))  # Separate weights for each channel
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        # pos is expected to be of shape [n, 3, n_vector_features]
        norm = torch.norm(pos, dim=1, keepdim=True)  # Normalize over the 3 dimension
        batch_size = int(batch.max()) + 1
        mean_norm = scatter_mean(norm, batch, dim=0, dim_size=batch_size)
        new_pos = self.weight * pos / (mean_norm[batch] + self.eps)
        return new_pos


class MultidimEquiUpdate(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim, n_vector_features=16):
        super().__init__()
        self.coord_norm = MultidimE3Norm(n_vector_features)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, hidden_dim * 2))
        input_ch = hidden_dim * 2 + edge_dim + dist_dim
        self.input_lin = nn.Linear(input_ch, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, n_vector_features, bias=False)
        )
        self.tanh = nn.GELU(approximate='tanh')

    def forward(self, h, pos, edge_index, edge_attr, dist, time_emb, batch_index):
        row, col = edge_index
        h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
        coord_diff = pos[row] - pos[col]
        coord_diff = self.coord_norm(coord_diff, batch_index[row])

        shift, scale = self.time_mlp(time_emb).chunk(2, dim=1)
        inv = modulate(self.ln(self.input_lin(h_input)), shift, scale)
        inv = self.tanh(self.coord_mlp(inv))

        # aggregate position
        trans = coord_diff * inv.unsqueeze(1)
        agg = scatter(trans, row, 0, reduce='add', dim_size=pos.size(0))
        pos = pos + agg

        return pos


class MultiEdgeMixInLayer(MessagePassing):
    def __init__(
        self,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
        n_vector_features=128,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")  #! This should be target to source
        # self.pre_edge = MLP(invariant_edge_feat_dim, invariant_edge_feat_dim, invariant_edge_feat_dim)
        # self.edge_lin = MLP(2 * invariant_edge_feat_dim + 3, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.message_input_size = (
            invariant_node_feat_dim + invariant_node_feat_dim + n_vector_features + invariant_edge_feat_dim
        )
        self.phi_message = MLP(self.message_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.message_gate = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1, last_act="sigmoid")
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)
        self.h_input_size = 2 * invariant_node_feat_dim
        self.phi_h = MLP(self.h_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.coor_update_clamp_value = 10.0
        # self.reset_parameters()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.use_cross_product = True
        if self.use_cross_product:
            self.phi_x_cross = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)
        self.x_norm = MultidimE3Norm(n_vector_features)

    #     self.apply(self.init_)

    # # def reset_parameters(self):
    # def init_(self, module): #! this made it worse
    #     if type(module) in {nn.Linear}:
    #         # seems to be needed to keep the network from exploding to NaN with greater depths
    #         nn.init.xavier_normal_(module.weight)
    #         nn.init.zeros_(module.bias)

    def forward(self, batch, X, H, edge_index, edge_attr):
        # import ipdb; ipdb.set_trace()
        # X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])
        # X = center_x(X, batch)
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
        edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)
        m_ij = self.phi_message(torch.cat([H[target], H[source], edge_attr_feat], dim=-1))
        coor_wij = self.phi_x(m_ij)  # E x 3
        if self.coor_update_clamp_value:
            coor_wij.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist.unsqueeze(1) + 1e-8))
        x_update = scatter(X_rel_norm * coor_wij.unsqueeze(-1), index=target, dim=0, reduce='sum', dim_size=X.shape[0])
        X_out = X + x_update
        if self.use_cross_product:
            mean = scatter(X, index=batch, dim=0, reduce='mean', dim_size=X.shape[0])
            x_src = X[source] - mean[source]
            x_tgt = X[target] - mean[target]
            cross = torch.cross(x_src, x_tgt, dim=1)
            cross = cross / (1 + torch.linalg.norm(cross, dim=1, keepdim=True))
            coor_wij_cross = self.phi_x_cross(m_ij)
            if self.coor_update_clamp_value:
                coor_wij_cross.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
            x_update_cross = scatter(
                cross * coor_wij_cross.unsqueeze(-1), index=target, dim=0, reduce='sum', dim_size=X.shape[0]
            )
            X_out = X_out + x_update_cross

        # m_i, m_ij = self.aggregate(inputs = m_ij * self.message_gate(m_ij), index = i, dim_size = X.shape[0])
        m_i = scatter(
            m_ij * self.message_gate(m_ij), index=target, dim=0, reduce='sum', dim_size=X.shape[0]
        )  #! Sigmoid over the gate matters a lot
        H_out = H + self.phi_h(torch.cat([H, m_i], dim=-1))  # self.h_norm(H, batch)
        return X_out, H_out


class MultiEdgeMixOutLayer(MessagePassing):
    def __init__(
        self,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
        n_vector_features=128,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")  #! This should be target to source
        self.pre_edge = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_edge_feat_dim)
        # self.edge_lin = MLP(2 * invariant_edge_feat_dim + 3, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.message_input_size = invariant_node_feat_dim + invariant_node_feat_dim + n_vector_features
        self.phi_message = MLP(self.message_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.message_gate = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1, last_act="sigmoid")
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)
        self.h_input_size = 2 * invariant_node_feat_dim
        self.phi_h = MLP(self.h_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.coor_update_clamp_value = 10.0
        # self.reset_parameters()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.use_cross_product = True
        if self.use_cross_product:
            self.phi_x_cross = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)
        self.x_norm = MultidimE3Norm(n_vector_features)

    #     self.apply(self.init_)

    # # def reset_parameters(self):
    # def init_(self, module): #! this made it worse
    #     if type(module) in {nn.Linear}:
    #         # seems to be needed to keep the network from exploding to NaN with greater depths
    #         nn.init.xavier_normal_(module.weight)
    #         nn.init.zeros_(module.bias)

    def forward(self, batch, X, H, edge_index):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
        edge_attr_feat = rel_dist  # torch.cat([edge_attr, rel_dist], dim=-1)
        m_ij = self.phi_message(torch.cat([H[target], H[source], edge_attr_feat], dim=-1))
        coor_wij = self.phi_x(m_ij)  # E x 1
        if self.coor_update_clamp_value:
            coor_wij.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist.unsqueeze(1) + 1e-8))
        x_update = scatter(X_rel_norm * coor_wij.unsqueeze(-1), index=target, dim=0, reduce='sum', dim_size=X.shape[0])
        X_out = X + x_update
        if self.use_cross_product:
            mean = scatter(X, index=batch, dim=0, reduce='mean', dim_size=X.shape[0])
            x_src = X[source] - mean[source]
            x_tgt = X[target] - mean[target]
            cross = torch.cross(x_src, x_tgt, dim=1)
            cross = cross / (1 + torch.linalg.norm(cross, dim=1, keepdim=True))
            coor_wij_cross = self.phi_x_cross(m_ij)
            if self.coor_update_clamp_value:
                coor_wij_cross.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
            x_update_cross = scatter(
                cross * coor_wij_cross.unsqueeze(-1), index=target, dim=0, reduce='sum', dim_size=X.shape[0]
            )
            X_out = X_out + x_update_cross

        # m_i, m_ij = self.aggregate(inputs = m_ij * self.message_gate(m_ij), index = i, dim_size = X.shape[0])
        m_i = scatter(
            m_ij * self.message_gate(m_ij), index=target, dim=0, reduce='sum', dim_size=X.shape[0]
        )  #! Sigmoid over the gate matters a lot
        H_out = H + self.phi_h(torch.cat([H, m_i], dim=-1))

        edge_attr = self.pre_edge(m_ij)
        return X_out, H_out, edge_attr


class MultiBondRefine(MessagePassing):
    def __init__(self, invariant_node_feat_dim=64, invariant_edge_feat_dim=32, n_vector_features=128):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")
        # self.x_norm = E3Norm()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.edge_norm = BatchLayerNorm(invariant_edge_feat_dim)
        self.bond_norm = BatchLayerNorm(invariant_edge_feat_dim)
        in_feats = 2 * invariant_node_feat_dim + n_vector_features + invariant_edge_feat_dim
        self.refine_layer = torch.nn.Sequential(
            torch.nn.Linear(in_feats, invariant_edge_feat_dim),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(invariant_edge_feat_dim, invariant_edge_feat_dim),
        )

    def forward(self, batch, X, H, edge_index, edge_attr):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])
        # X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  # E
        edge_attr = self.edge_norm(edge_attr, edge_batch)
        infeats = torch.cat([H[target], H[source], rel_dist, edge_attr], dim=-1)
        return self.bond_norm(self.refine_layer(infeats), edge_batch)


class MultiXEGNN(MessagePassing):
    """
    X only EGNN
    """

    def __init__(self, invariant_node_feat_dim=64, n_vector_features=128):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")  #! This should be target to source
        self.message_input_size = 2 * invariant_node_feat_dim + n_vector_features  # + invariant_edge_feat_dim
        self.phi_message = MLP(self.message_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)
        self.coor_update_clamp_value = 10.0
        # self.reset_parameters()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.use_cross_product = True
        if self.use_cross_product:
            self.phi_x_cross = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)
        self.x_norm = MultidimE3Norm(n_vector_features)
        # TODO: @Ali what is good weight inititalization for EGNN?

    #     self.apply(self.init_)

    # # def reset_parameters(self):
    # def init_(self, module): #! this made it worse
    #     if type(module) in {nn.Linear}:
    #         # seems to be needed to keep the network from exploding to NaN with greater depths
    #         nn.init.xavier_normal_(module.weight)
    #         nn.init.zeros_(module.bias)

    def forward(self, batch, X, H, edge_index, edge_attr=None):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
        if edge_attr is not None:
            edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feat = rel_dist
        m_ij = self.phi_message(torch.cat([H[target], H[source], edge_attr_feat], dim=-1))
        coor_wij = self.phi_x(m_ij)  # E x 3
        if self.coor_update_clamp_value:
            coor_wij.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
        # import ipdb; ipdb.set_trace()
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist.unsqueeze(1) + 1e-8))
        x_update = scatter(X_rel_norm * coor_wij.unsqueeze(-1), index=target, dim=0, reduce='sum', dim_size=X.shape[0])
        X_out = X + x_update
        if self.use_cross_product:
            mean = scatter(X, index=batch, dim=0, reduce='mean', dim_size=X.shape[0])
            x_src = X[source] - mean[source]
            x_tgt = X[target] - mean[target]
            cross = torch.cross(x_src, x_tgt, dim=1)
            cross = cross / (1 + torch.linalg.norm(cross, dim=1, keepdim=True))
            coor_wij_cross = self.phi_x_cross(m_ij)
            if self.coor_update_clamp_value:
                coor_wij_cross.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
            x_update_cross = scatter(
                cross * coor_wij_cross.unsqueeze(-1), index=target, dim=0, reduce='sum', dim_size=X.shape[0]
            )
            X_out = X_out + x_update_cross

        H_out = H
        return X_out, H_out


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
    num_classes = 13
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
    H = atom_embedder(F.one_hot(ligand_feats, num_classes).float())
    A = adj_matrix
    mask = batch_ligand.unsqueeze(1) == batch_ligand.unsqueeze(0)  # Shape: (75, 75)
    E_idx = mask.nonzero(as_tuple=False).t()
    self_loops = E_idx[0] != E_idx[1]
    E_idx = E_idx[:, self_loops]
    Z = atom_embedder(F.one_hot(ligand_feats, num_classes).float()).unsqueeze(1) * atom_embedder(
        F.one_hot(ligand_feats, num_classes).float()
    ).unsqueeze(0)

    import ipdb

    ipdb.set_trace()
    time = torch.tensor([0.2, 0.4, 0.6, 0.8])
    # temb = TimestepEmbedder(64)(time, batch_ligand)
    # dit = DiTBlock(64, 8)
    # out = dit(batch_ligand, H, temb[batch_ligand], Z)
    print("Success")
