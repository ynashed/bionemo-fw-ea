# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
from torch_scatter import scatter, scatter_mean, scatter_sum
from torch_sparse import SparseTensor


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


class E3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((1, 1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ):
        norm = torch.norm(pos, dim=-1, keepdim=True)
        batch_size = int(batch.max()) + 1
        mean_norm = scatter_mean(norm, batch, dim=0, dim_size=batch_size)
        new_pos = self.weight * pos / (mean_norm[batch] + self.eps)
        return new_pos


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        activation: str = 'silu',
        dropout: float = 0.0,
        last_act: str = None,
    ):
        """
        Initialize the MLP.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_size (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output features.
            num_hidden_layers (int): Number of hidden layers.
            activation (str): Activation function to use ('relu', 'silu', etc.).
            dropout (float): Dropout probability (between 0 and 1).
        """
        super(MLP, self).__init__()

        if activation not in NONLINEARITIES:
            raise ValueError(f"Activation function must be one of {list(NONLINEARITIES.keys())}")

        self.act_layer = NONLINEARITIES[activation]

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(NONLINEARITIES[activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(NONLINEARITIES[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))
        if last_act:
            layers.append(NONLINEARITIES[last_act])

        # Combine all layers into a sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, batch=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        if batch is not None:
            return t_emb[batch]
        else:
            return t_emb


class AdaLN(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int):
        """
        Initialize the Adaptive Layer Normalization (AdaLN) module.
        This implementation does not learn a gate.

        Args:
            condition_dim (int): Dimension of the conditional input.
            feature_dim (int): Dimension of the input features.
        """
        super().__init__()
        self.layernorm = BatchLayerNorm(feature_dim)
        self.scale_shift_mlp = MLP(condition_dim, 2 * feature_dim, 2 * feature_dim)

    def forward(self, h: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AdaLN module.

        Args:
            h (torch.Tensor): Input tensor to be normalized (batch_size, feature_dim).
            t (torch.Tensor): Conditional input tensor (batch_size, condition_dim).

        Returns:
            torch.Tensor: Normalized output tensor (batch_size, feature_dim).
        """
        # scale = 1 + self.scale_mlp(t)
        # shift = self.shift_mlp(t)
        scale, shift = self.scale_shift_mlp(t).chunk(2, dim=-1)
        return (1 + scale[batch]) * self.layernorm(h, batch) + shift[batch]


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def get_triplet(edge_index: torch.Tensor, num_nodes: int):
    """
    Compute triplets of nodes and corresponding edge indices in a graph.

    Args:
        edge_index (torch.Tensor): The edge index tensor representing
        the connections between source and target nodes.
        num_nodes (int): The total number of nodes in the graph.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing:
            - input_edge_index (torch.Tensor): The input edge index tensor.
            - idx_i (torch.Tensor): Node indices i in the triplets (k->j->i).
            - idx_j (torch.Tensor): Node indices j in the triplets (k->j->i).
            - idx_k (torch.Tensor): Node indices k in the triplets (k->j->i).
            - idx_kj (torch.Tensor): Edge indices (k-j) in the triplets.
            - idx_ji (torch.Tensor): Edge indices (j->i) in the triplets.
    """
    assert edge_index.size(0) == 2
    input_edge_index = edge_index.clone()
    source, target = edge_index  # j->i
    # create identifiers for edges based on (source, target)
    value = torch.arange(source.size(0), device=source.device)
    # as row-index select the target (column) nodes --> transpose
    # create neighbours from j
    adj_t = SparseTensor(row=target, col=source, value=value, sparse_sizes=(num_nodes, num_nodes))
    # get neighbours from j
    adj_t_row = adj_t[source]
    # returns the target nodes (k) that include the source (j)
    # note there can be path i->j->k where k is i
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
    # print(num_triplets)
    # Node indices (k->j->i) for triplets.
    idx_i = target.repeat_interleave(num_triplets)
    idx_j = source.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()  # get index for k
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    # print(idx_i); print(idx_j); print(idx_k)
    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return input_edge_index, idx_i, idx_j, idx_k, idx_kj, idx_ji


class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian=0, update_x=True, act_fn='silu', norm=False):
        super().__init__()
        self.r_min = 0.0
        self.r_max = 10.0
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm

        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + 1, hidden_dim, hidden_dim, activation=act_fn)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            # self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]
            layer = nn.Linear(hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            x_mlp.append(layer)
            x_mlp.append(nn.Tanh())
            self.x_mlp = nn.Sequential(*x_mlp)

        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, activation=act_fn)

    def forward(self, h, x, edge_index, edge_attr=None):
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        rel_x = x[dst] - x[src]
        d_sq = torch.sum(rel_x**2, -1, keepdim=True)

        d_feat = d_sq
        if edge_attr is not None:
            edge_feat = torch.cat([d_feat, edge_attr], -1)
        else:
            edge_feat = d_sq

        mij = self.edge_mlp(torch.cat([hi, hj, edge_feat], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        h = h + self.node_mlp(torch.cat([mi, h], -1))
        if self.update_x:
            # x update in Eq(4)
            xi, xj = x[dst], x[src]
            # (xi - xj) / (\|xi - xj\| + C) to make it more stable
            delta_x = scatter_sum((xi - xj) / (torch.sqrt(d_sq + 1e-8) + 1) * self.x_mlp(mij), dst, dim=0)
            x = x + delta_x  # * mask_ligand[:, None]  # only ligand positions will be updated

        return h, x


class EquivariantMessagePassingLayer(MessagePassing):
    def __init__(
        self,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")  #! This should be target to source
        self.pre_edge = MLP(invariant_edge_feat_dim, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.edge_lin = MLP(2 * invariant_edge_feat_dim + 3, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.message_input_size = invariant_node_feat_dim + invariant_node_feat_dim + 1 + invariant_edge_feat_dim
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
        self.x_norm = E3Norm()

    #     self.apply(self.init_)

    # # def reset_parameters(self):
    # def init_(self, module): #! this made it worse
    #     if type(module) in {nn.Linear}:
    #         # seems to be needed to keep the network from exploding to NaN with greater depths
    #         nn.init.xavier_normal_(module.weight)
    #         nn.init.zeros_(module.bias)

    def mix_edges(self, batch, X, E_idx, E, k=4):
        num_nodes = X.size(0)
        A_full = torch.zeros(
            size=(num_nodes, num_nodes, E.size(-1)),
            device=E.device,
            dtype=E.dtype,
        )  # N x N x num classes = 5
        A_full[E_idx[0], E_idx[1], :] = E

        # create kNN graph
        edge_index_knn = knn_graph(x=X, k=k, batch=batch, flow="source_to_target")  # 2 x KE
        j, i = edge_index_knn
        p_ij = X[j] - X[i]
        p_ij_n = torch.nn.functional.normalize(p_ij, p=2, dim=-1)
        d_ij = torch.pow(p_ij, 2).sum(-1, keepdim=True).sqrt()

        edge_ij = A_full[j, i, :]

        edge_index_knn, _, _, _, idx_kj, idx_ji = get_triplet(
            edge_index_knn, num_nodes=num_nodes
        )  # 2 x KE,  M, M, M, M, M
        p_jk = -1.0 * p_ij_n[idx_kj]
        p_ji = p_ij_n[idx_ji]
        theta_ijk = torch.arccos(torch.sum(p_jk * p_ji, -1, keepdim=True).clamp_(-1.0 + 1e-7, 1.0 - 1e-7))  # M x 1
        d_ji = d_ij[idx_ji]  # M x 1
        d_jk = d_ij[idx_kj]  # M x 1
        edge_0 = edge_ij[idx_ji]  # M x 5
        edge_1 = edge_ij[idx_kj]  # M x 5
        f_ijk = torch.cat([edge_0, edge_1, theta_ijk, d_ji, d_jk], dim=-1)  # M x 2*classes + 3 = 13
        f_ijk = self.edge_lin(f_ijk)  # M x num classes

        aggr_edges = scatter(
            src=f_ijk,
            index=idx_ji,
            dim=0,
            reduce="mean",
            dim_size=edge_index_knn.size(-1),
        )  # KE x num_classes = 5
        A_aggr = torch.zeros_like(A_full)  # N x N x 5
        A_aggr[edge_index_knn[0], edge_index_knn[1], :] = aggr_edges
        A_out = A_full + A_aggr  # N x N x 5
        E = A_out[E_idx[0], E_idx[1], :]  # E x 5
        return E

    def forward(self, batch, X, H, edge_index, edge_attr):
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        edge_attr = self.mix_edges(batch, X, edge_index, self.pre_edge(edge_attr), k=4)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)
        # m_i, m_ij = self.propagate(edge_index=edge_index, X=X, X_rel=rel_coors, H=H, edge_attr=edge_attr_feat, batch=batch)
        #
        m_ij = self.phi_message(torch.cat([H[target], H[source], edge_attr_feat], dim=-1))
        coor_wij = self.phi_x(m_ij)  # E x 3
        if self.coor_update_clamp_value:
            coor_wij.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist + 1e-8))
        x_update = scatter(X_rel_norm * coor_wij, index=target, dim=0, reduce='sum', dim_size=X.shape[0])
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
            x_update_cross = scatter(cross * coor_wij_cross, index=target, dim=0, reduce='sum', dim_size=X.shape[0])
            X_out = X_out + x_update_cross

        # m_i, m_ij = self.aggregate(inputs = m_ij * self.message_gate(m_ij), index = i, dim_size = X.shape[0])
        m_i = scatter(
            m_ij * self.message_gate(m_ij), index=target, dim=0, reduce='sum', dim_size=X.shape[0]
        )  #! Sigmoid over the gate matters a lot
        # import ipdb; ipdb.set_trace()
        H_out = H + self.phi_h(torch.cat([H, m_i], dim=-1))  # self.h_norm(H, batch)
        # H_out = self.h_norm(
        # H + self.phi_h(torch.cat([H, m_i], dim=-1)), batch
        # )  # self.h_norm(H, batch) #! the use of LN here prevents H blow up similar to FlowMol
        #! is the fact that FABind uses attention so a softmax weight (0,1) to scale before doing the residual update prevent explosion?
        #! We use target as the index since we are aaggregating over i [1, 0] and [2, 0] we want to argegat over the 0 so we use its edge index
        return X_out, H_out, edge_attr, m_ij

    # def message(self, H_i, H_j, edge_attr):
    #     # edge_attr already has the norm of distances
    #     mij = self.phi_message(torch.cat([H_i, H_j, edge_attr], dim=-1))  # E x D
    #     return mij

    # def aggregate(
    #     self,
    #     inputs,
    #     index,
    #     dim_size=None,
    # ):
    #     # import ipdb; ipdb.set_trace()
    #     # mi = scatter(inputs, index=index, dim=0, reduce="add", dim_size=dim_size) # index here is target
    #     # return mi, inputs
    #     return inputs

    #  def coord2radial(self, coord, edge_index):
    #     row, col = edge_index
    #     coord_diff = coord[row] - coord[col]
    #     radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))
    #     return radial, coord_diff

    # def coord2diff(self, x, edge_index, norm=False, norm_constant=1):
    #     row, col = edge_index
    #     coord_diff = x[row] - x[col]
    #     radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    #     norm = torch.sqrt(radial + 1e-8)
    #     if norm:
    #         coord_diff = coord_diff / (norm + norm_constant)
    #     return radial, coord_diff

    # def forward(self, h, x, edge_index, edge_attr, mask):
    #     src, dst = edge_index
    #     hi, hj = h_i, h_j
    #     radial, coord_diff = self.coord2radial(x, edge_index)
    #     radial = radial.reshape(radial.shape[0], -1)
    #     mij = self.phi_message(torch.cat([hi, hj, radial, edge_attr], -1))
    #     gij = self.message_gate(mij)
    #     h_agg = unsorted_segment_sum(mij * gij, src, num_segments=x.size(0))
    #     h = h + self.phi_h(torch.cat([x, h_agg, h], -1))
    #     trans = coord_diff * self.phi_x(mij).unsqueeze(-1)
    #     x_agg = unsorted_segment_mean(trans, src, num_segments=x.size(0))
    #     x = x + x_agg.clamp(-self.coord_change_maximum, self.coord_change_maximum)
    #     return h, x


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
                # import ipdb; ipdb.set_trace()
                adj_matrix[idx][jdx] = torch.nn.functional.one_hot(torch.randint(0, 5, (1,)), 5).squeeze(0)
    # print(adj_matrix)

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
    source, target = E_idx
    E = A[source, target]

    source, target = E_idx
    r = X[target] - X[source]  # E x 3
    a = X[target] * X[source]
    a = a.sum(-1)  # E
    d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
    d = d.sqrt()  # E
    r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))  # E x 3
    E = A[source, target]  # E x 5
    E_all = torch.cat((d.unsqueeze(1), a.unsqueeze(1), r_norm, E), dim=-1)  # E x 10
    edge_embedder = nn.Linear(5, 32)
    #! Need to hit E with MLP first
    E = edge_embedder(E.float())

    # import ipdb; ipdb.set_trace()
    model = EquivariantMessagePassingLayer()  #! Layer norm forces stable.
    print(X.sum(), H.sum(), E.sum())
    for i in range(25):
        X, H, E, m_ij = model(batch_ligand, X, H, E_idx, E)
        print(X.sum(), H.sum(), E.sum())

    # model = EnBaseLayer(64, 5) #! Stable but increasing
    # for i in range(25):
    #     H, X = model(H, X, E_idx, E)
    #     print(X.sum(), H.sum(), E.sum())
    # import ipdb; ipdb.set_trace()
    print("Success")
