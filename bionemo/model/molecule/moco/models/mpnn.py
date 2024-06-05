import math

import torch
import torch.nn as nn
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm


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


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        activation: str = 'silu',
        dropout: float = 0.0,
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

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
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


class EGNN(nn.Module):
    def __init__(self, invariant_feature_dim, equivariant_feature_dim):
        self.time_embedder = TimestepEmbedder(invariant_feature_dim)
        self.adaLN = AdaLN(invariant_feature_dim, invariant_feature_dim)
        self.message_input_size = 0
        self.h_input_size = 0
        self.phi_message = MLP(self.message_input_size, invariant_feature_dim, invariant_feature_dim)
        self.message_gate = MLP(invariant_feature_dim, invariant_feature_dim, 1)
        self.phi_x = MLP(invariant_feature_dim, invariant_feature_dim, equivariant_feature_dim)
        self.phi_h = MLP(self.h_input_size, invariant_feature_dim, invariant_feature_dim)
        self.self.node_agg_scale = 1  # /100
        self.coord_change_maximum = 10
        # TODO: Understand how to do all to all message passing use EqGatDiff as a reference https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/experiments/diffusion_discrete_addfeats.py#L43
        # ! This is the underlying MPNN they use EQGATGlobalEdgeConvFinal

    def coord2radial(self, coord, edge_index):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))
        return radial, coord_diff

    def coord2diff(self, x, edge_index, norm=False, norm_constant=1):
        row, col = edge_index
        coord_diff = x[row] - x[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        norm = torch.sqrt(radial + 1e-8)
        if norm:
            coord_diff = coord_diff / (norm + norm_constant)
        return radial, coord_diff

    def forward(self, h, x, edge_index, edge_attr, mask):
        src, dst = edge_index
        hi, hj = h[src], h[dst]
        radial, coord_diff = self.coord2radial(x, edge_index)
        radial = radial.reshape(radial.shape[0], -1)
        mij = self.phi_message(torch.cat([hi, hj, radial, edge_attr], -1))
        gij = self.message_gate(mij)
        h_agg = unsorted_segment_sum(mij * gij, src, num_segments=x.size(0))
        h = h + self.phi_h(torch.cat([x, h_agg, h], -1))
        trans = coord_diff * self.phi_x(mij).unsqueeze(-1)
        x_agg = unsorted_segment_mean(trans, src, num_segments=x.size(0))
        x = x + x_agg.clamp(-self.coord_change_maximum, self.coord_change_maximum)
        return h, x


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

    atom_embedder = nn.Linear(1, 64)
    X = ligand_pos
    H = atom_embedder(ligand_feats.unsqueeze(1).float())
    A = adj_matrix
    mask = batch_ligand.unsqueeze(1) == batch_ligand.unsqueeze(0)  # Shape: (75, 75)
    E_idx = mask.nonzero(as_tuple=False).t()
    self_loops = E_idx[0] != E_idx[1]
    E_idx = E_idx[:, self_loops]
    Z = atom_embedder(ligand_feats.unsqueeze(1).float()).unsqueeze(1) * atom_embedder(
        ligand_feats.unsqueeze(1).float()
    ).unsqueeze(0)

    #! this is used if self.bond pred is true but its not
    # D = (X[E_idx[1]] - X[E_idx[0]]).pow(2).sum(-1, keepdim=True).sqrt() # E x 1
    # E_idx = torch.concat([H[E_idx[1]] + H[E_idx[0]], D], dim=-1) # E x 64 + 1
    from torch_sparse import SparseTensor

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

    import ipdb

    ipdb.set_trace()
    source, target = E_idx
    r = X[target] - X[source]  # E x 3
    a = X[target] * X[source]
    a = a.sum(-1)  # E
    d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
    d = d.sqrt()  # E
    r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))  # E x 3
    E = A[source, target]  # E x 5
    E = torch.cat((d.unsqueeze(1), a.unsqueeze(1), r_norm, E), dim=-1)  # E x 10
    import ipdb

    ipdb.set_trace()
    # Edge Message Passing
    from torch_geometric.nn import knn_graph

    num_nodes = X.size(0)

    E_full = torch.zeros(
        size=(num_nodes, num_nodes, E.size(-1)),
        device=E.device,
        dtype=E.dtype,
    )
    E_full[E_idx[0], E_idx[1], :] = E

    # create kNN graph
    edge_index_knn = knn_graph(x=X, k=4, batch=batch_ligand, flow="source_to_target")
    j, i = edge_index_knn
    p_ij = X[j] - X[i]
    p_ij_n = torch.nn.functional.normalize(p_ij, p=2, dim=-1)
    d_ij = torch.pow(p_ij, 2).sum(-1, keepdim=True).sqrt()

    edge_ij = E_full[j, i, :]

    edge_index_knn, idx_i, idx_j, idx_k, idx_kj, idx_ji = get_triplet(edge_index_knn, num_nodes=num_nodes)
