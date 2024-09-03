import functools
import math
import time

import einops
import torch

# import torch.cuda.nvtx as nvtx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_mean


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
    def __init__(self, n_vector_features: int = 1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        if n_vector_features > 1:
            self.weight = nn.Parameter(torch.ones((1, 1, n_vector_features)))  # Separate weights for each channel
        else:
            self.weight = nn.Parameter(torch.ones((1, 1)))
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
        bias: bool = True,
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
        layers.append(nn.Linear(input_dim, hidden_size, bias=bias))
        layers.append(NONLINEARITIES[activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        if num_hidden_layers > 0:
            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
                layers.append(NONLINEARITIES[activation])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim, bias=bias))
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
        scale, shift = self.scale_shift_mlp(t).chunk(2, dim=-1)
        return (1 + scale[batch]) * self.layernorm(h, batch) + shift[batch]


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#! Below are taken from ESM3 for now
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


class PredictionHead(nn.Module):
    def __init__(self, num_classes, feat_dim, discrete=True, edge_prediction=False, distance_prediction=False):
        super().__init__()
        self.num_classes = num_classes
        self.discrete = discrete
        self.projection = MLP(feat_dim, feat_dim, num_classes)
        if edge_prediction:
            self.post_process = MLP(num_classes, num_classes, num_classes)
        if distance_prediction:
            self.post_process = MLP(feat_dim, feat_dim, num_classes, last_act="sigmoid")
            self.embedding = MLP(feat_dim, feat_dim, feat_dim)

    #! Even if we have a masking state we still predict it but always mask it even on the forward pass as done in MultiFlow. The loss is taken over teh logits so its not masked
    def forward(self, batch, H):
        logits = self.projection(H)
        if self.discrete:
            probs = F.softmax(logits, dim=-1)  # scatter_softmax(logits, index=batch, dim=0, dim_size=H.size(0))
        else:
            probs = H - scatter_mean(H, index=batch, dim=0)[batch]
        return logits, probs

    def predict_edges(self, batch, E, E_idx):
        #! EQGAT also uses hi hj and Xi-Xj along with f(E) see coordsatomsbonds.py line 121 https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/e3moldiffusion/coordsatomsbonds.py#L121C32-L121C44
        # import ipdb; ipdb.set_trace()
        E = self.projection(E)
        src, dst = E_idx
        N = batch.size(0)
        e_dense = torch.zeros(N, N, E.size(-1), device=E.device)
        e_dense[src, dst, :] = E
        e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
        e = e_dense[src, dst, :]  # E x 5
        logits = self.post_process(e)  # E x 5
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def predict_distances(self, batch, Z):
        input = Z + Z.permute(1, 0, 2)
        logits = self.post_process(input) * self.embedding(input)
        logits = self.projection(logits).squeeze(-1)
        return logits


class BondRefine(nn.Module):
    def __init__(
        self,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
    ):
        super().__init__()
        # self.x_norm = E3Norm()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.edge_norm = BatchLayerNorm(invariant_edge_feat_dim)
        self.bond_norm = BatchLayerNorm(invariant_edge_feat_dim)
        in_feats = 2 * invariant_node_feat_dim + 1 + invariant_edge_feat_dim
        self.refine_layer = torch.nn.Sequential(
            torch.nn.Linear(in_feats, invariant_edge_feat_dim),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(invariant_edge_feat_dim, invariant_edge_feat_dim),
        )

    def forward(self, batch, X, H, edge_index, edge_attr):
        X = X - scatter_mean(X, index=batch, dim=0)[batch]
        # X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        # edge_batch, counts = torch.unique(batch, return_counts=True)
        # edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  # E
        edge_batch = batch[source]
        edge_attr = self.edge_norm(edge_attr, edge_batch)
        infeats = torch.cat([H[target], H[source], rel_dist, edge_attr], dim=-1)
        return self.bond_norm(self.refine_layer(infeats), edge_batch)


class EquivariantBlock(nn.Module):
    """
    X only EGNN-like architecture
    """

    def __init__(
        self, invariant_node_feat_dim=64, invariant_edge_feat_dim=64, n_vector_features=128, use_cross_product=False
    ):
        super().__init__()  #! This should be target to source
        self.message_input_size = (
            2 * invariant_node_feat_dim + n_vector_features + invariant_edge_feat_dim + invariant_node_feat_dim
        )
        self.phi_message = MLP(self.message_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, n_vector_features)
        self.coor_update_clamp_value = 10.0
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.use_cross_product = use_cross_product
        if self.use_cross_product:
            self.phi_x_cross = MLP(invariant_node_feat_dim, invariant_node_feat_dim, n_vector_features)
        self.x_norm = E3Norm(n_vector_features)
        # TODO: @Ali what is good weight inititalization for EGNN?

    #     self.apply(self.init_)

    # # def reset_parameters(self):
    # def init_(self, module): #! this made it worse
    #     if type(module) in {nn.Linear}:
    #         # seems to be needed to keep the network from exploding to NaN with greater depths
    #         nn.init.xavier_normal_(module.weight)
    #         nn.init.zeros_(module.bias)

    def forward(self, batch, X, H, edge_index, edge_attr=None, te=None):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])[batch]
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index

        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
        if edge_attr is not None:
            edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feat = rel_dist
        m_ij = self.phi_message(torch.cat([H[target], H[source], edge_attr_feat, te], dim=-1))

        coor_wij = self.phi_x(m_ij)  # E x 3
        if self.coor_update_clamp_value:
            coor_wij.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist.unsqueeze(1) + 1e-8))
        x_update = scatter(X_rel_norm * coor_wij.unsqueeze(1), index=target, dim=0, reduce='sum', dim_size=X.shape[0])
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
                cross * coor_wij_cross.unsqueeze(1), index=target, dim=0, reduce='sum', dim_size=X.shape[0]
            )
            X_out = X_out + x_update_cross

        return X_out


class DiTeBlock(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(
        self,
        hidden_size,
        edge_hidden_size,
        num_heads,
        mlp_expansion_ratio=4.0,
        use_z=None,
        n_vector_features=128,
        scale_dist_features=4,
        dropout=0.0,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.pre_attn_norm = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.pre_attn_norm_edge = BatchLayerNorm(edge_hidden_size, affine=False, eps=1e-6)
        #! AlphaFold3 has the scale as linear then sigmoid
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_edge_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(edge_hidden_size, 6 * edge_hidden_size, bias=True)
        )

        if scale_dist_features > 1:
            dist_size = n_vector_features * scale_dist_features
        else:
            dist_size = n_vector_features
        self.feature_embedder = MLP(hidden_size + hidden_size + edge_hidden_size + dist_size, hidden_size, hidden_size)

        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        # self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.d_head = hidden_size // num_heads
        self.use_z = use_z
        if use_z is not None:
            if use_z == "pair_embedding":
                self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, num_heads, bias=False))
            else:
                self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(dist_size, num_heads, bias=False))
            self.z_update = nn.Linear(hidden_size, hidden_size, bias=False)

        self.lin_edge0 = nn.Linear(hidden_size, edge_hidden_size, bias=False)
        self.lin_edge1 = nn.Linear(
            edge_hidden_size + n_vector_features * scale_dist_features, edge_hidden_size, bias=False
        )
        self.residual_ffn_norm = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.residual_ffn_norm_edge = BatchLayerNorm(edge_hidden_size, affine=False, eps=1e-6)
        self.ffn_edge = swiglu_ffn(edge_hidden_size, mlp_expansion_ratio, bias=False)

    def forward(
        self,
        batch: torch.Tensor,
        x: torch.Tensor,
        t_emb_h: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        t_emb_e: torch.Tensor = None,
        dist: torch.Tensor = None,
        Z: torch.Tensor = None,
    ):
        """
        This assume pytorch geometric batching so batch size of 1 so skip rotary as it depends on having an actual batch

        batch: N
        x: N x 256
        temb: N x 256
        edge_attr: E x 256
        edge_index: 2 x E
        """
        src, tgt = edge_index
        edge_batch = batch[src]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb_h).chunk(6, dim=1)
        (
            edge_shift_msa,
            edge_scale_msa,
            edge_gate_msa,
            edge_shift_mlp,
            edge_scale_mlp,
            edge_gate_mlp,
        ) = self.adaLN_edge_modulation(t_emb_e).chunk(6, dim=1)
        # Normalize x
        x_norm = modulate(self.pre_attn_norm(x, batch), shift_msa, scale_msa)
        edge_attr_norm = modulate(self.pre_attn_norm_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa)

        messages = self.feature_embedder(torch.cat([x_norm[src], x_norm[tgt], edge_attr_norm, dist], dim=-1))
        x_norm = scatter_mean(messages, src, dim=0)

        # Q K V Projection
        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.reshape(-1, self.num_heads, self.d_head)
        K = K.reshape(-1, self.num_heads, self.d_head)
        V = V.reshape(-1, self.num_heads, self.d_head)
        # Gather the query, key, and value tensors for the source and target nodes
        query_i = Q[src]  # [E, heads, d_head]
        key_j = K[tgt]  # [E, heads, d_head]
        value_j = V[tgt]  # [E, heads, d_head]
        # Compute the attention scores using dot-product attention mechanism
        alpha = query_i * key_j
        if self.use_z == 'pair_embedding':
            # import ipdb; ipdb.set_trace()
            alpha = alpha + self.pair_bias(Z).unsqueeze(-1)
        elif self.use_z == "distance":
            alpha = alpha + self.pair_bias(dist).unsqueeze(-1)
        alpha = alpha.sum(dim=-1) / math.sqrt(self.d_head)  # [E, heads]

        # Apply softmax to normalize attention scores across all edges directed to the same node
        alpha = softmax(alpha, tgt)  # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Apply dropout to attention scores

        # Multiply normalized attention scores with the value tensor to compute the messages
        msg = value_j * alpha.view(-1, self.num_heads, 1)  # [E, heads, d_head]
        # Aggregate messages to the destination nodes
        out = scatter(msg, tgt, dim=0, reduce='sum', dim_size=x.size(0))  # [N, heads, d_head]

        # Merge the heads and the output channels
        out = out.view(-1, self.num_heads * self.d_head)  # [N, heads * d_head]
        y = self.out_projection(out)

        # Gated Residual
        x = x + gate_msa * y
        # Feed Forward
        edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
        x = x + gate_mlp * self.ffn(modulate(self.residual_ffn_norm(x, batch), shift_mlp, scale_mlp))
        e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
        # import ipdb; ipdb.set_trace()
        edge_attr = edge + edge_gate_mlp * self.ffn_edge(
            modulate(self.residual_ffn_norm_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp)
        )
        if self.use_z == "pair_embedding":
            Z = Z + self.z_update(self.pyg_outer_update(batch, x))
            return x, edge_attr, Z
        else:
            return x, edge_attr, None

    def pyg_outer_update(self, batch, x):
        outer_products = []
        num_atoms = 0
        for molecule in range(max(batch.unique()) + 1):
            molecule_indices = (batch == molecule).nonzero(as_tuple=True)[0]
            molecule_features = x[molecule_indices]
            # Compute the outer product for the current molecule
            outer_product = molecule_features.unsqueeze(0) * molecule_features.unsqueeze(
                1
            )  # torch.einsum('ij,ik->ijk', molecule_features, molecule_features)
            outer_product = outer_product + outer_product.transpose(0, 1)
            outer_product = outer_product[~torch.eye(outer_product.size(0), dtype=torch.bool)].view(
                outer_product.size(0), -1
            )
            outer_products.append(outer_product.reshape(-1, molecule_features.shape[-1]))
            num_atoms += molecule_features.size(0)
        update = torch.cat(outer_products, dim=0)
        return update


def coord2distfn(x, edge_index, scale_dist_features=1, batch=None):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1)
    # import ipdb; ipdb.set_trace()
    if scale_dist_features >= 2:
        # dotproduct = x[row] * x[col]  # shape [num_edges, 3, k]
        # dotproduct = dotproduct.sum(-2)  # sum over the spatial dimension, shape [num_edges, k]
        # # Concatenate the computed features
        # radial = torch.cat([radial, dotproduct], dim=-1)  # shape [num_edges, 2*k]
        dotproduct = (x[row] * x[col]).sum(dim=-2, keepdim=False)  # shape [num_edges, 1]
        radial = torch.cat([radial, dotproduct], dim=-1)  # shape [num_edges, 2]
    if scale_dist_features == 4:
        p_i, p_j = x[edge_index[0]], x[edge_index[1]]
        d_i, d_j = (
            torch.pow(p_i, 2).sum(-2, keepdim=False).clamp(min=1e-6).sqrt(),
            torch.pow(p_j, 2).sum(-2, keepdim=False).clamp(min=1e-6).sqrt(),
        )
        radial = torch.cat([radial, d_i, d_j], dim=-1)

    return radial


class MoleculeDiffusion(nn.Module):
    def __init__(
        self,
        num_layers=10,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=22,
        edge_classes=5,
        num_heads=4,
        n_vector_features=128,
        scale_dist_features=4,
        use_z=None,
        attn_dropout=0.0,
        use_cross_product=False,
    ):
        super(MoleculeDiffusion, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)
        self.scale_dist_features = scale_dist_features

        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)

        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.use_z = use_z
        if use_z == "pair_embedding":
            self.Z_lin = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)

        for i in range(num_layers):
            self.dit_layers.append(
                DiTeBlock(
                    invariant_node_feat_dim,
                    invariant_edge_feat_dim,
                    num_heads,
                    mlp_expansion_ratio=4.0,
                    use_z=use_z,
                    n_vector_features=n_vector_features,
                    scale_dist_features=scale_dist_features,
                    dropout=attn_dropout,
                )
            )
            self.egnn_layers.append(
                EquivariantBlock(
                    invariant_node_feat_dim,
                    invariant_edge_feat_dim,
                    n_vector_features,
                    use_cross_product=use_cross_product,
                )
            )

        self.node_blocks = nn.ModuleList(
            [nn.Linear(invariant_node_feat_dim, invariant_node_feat_dim) for i in range(num_layers)]
        )
        self.edge_blocks = nn.ModuleList(
            [nn.Linear(invariant_edge_feat_dim, invariant_edge_feat_dim) for i in range(num_layers)]
        )

    def pyg_outer_update(self, batch, x):
        outer_products = []
        num_atoms = 0
        for molecule in range(max(batch.unique()) + 1):
            molecule_indices = (batch == molecule).nonzero(as_tuple=True)[0]
            molecule_features = x[molecule_indices]
            # Compute the outer product for the current molecule
            outer_product = molecule_features.unsqueeze(0) * molecule_features.unsqueeze(
                1
            )  # torch.einsum('ij,ik->ijk', molecule_features, molecule_features)
            outer_product = outer_product + outer_product.transpose(0, 1)
            outer_product = outer_product[~torch.eye(outer_product.size(0), dtype=torch.bool)].view(
                outer_product.size(0), -1
            )
            outer_products.append(outer_product.reshape(-1, molecule_features.shape[-1]))
            num_atoms += molecule_features.size(0)
        update = torch.cat(outer_products, dim=0)
        return update

    def forward(self, batch, X, H, E_idx, E, t_discrete, t_continuous=None):
        torch.max(batch) + 1
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K
        H = self.atom_embedder(H)
        E = self.edge_embedder(E)
        te = self.time_embedding(t_discrete)
        te_h = te[batch]
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))
        te_e = te[edge_batch]
        if t_continuous:
            te_continuous = self.time_embedding(t_continuous)
            te_e_cont = te_continuous[edge_batch]
        else:
            te_e_cont = te_e
        edge_attr = E
        if self.use_z == "pair_embedding":
            Z = self.pyg_outer_update(batch, H)
        else:
            Z = None

        atom_hids = H
        edge_hids = edge_attr

        for layer_index in range(len(self.dit_layers)):
            # with nvtx.range(f"DiTe Layer {layer_index}"):
            distances = coord2distfn(pos, E_idx, self.scale_dist_features, batch)  # E x K
            H, edge_attr, Z = self.dit_layers[layer_index](batch, H, te_h, edge_attr, E_idx, te_e, distances, Z)
            # with nvtx.range(f"EGNN Layer {layer_index}"):
            pos = self.egnn_layers[layer_index](batch, pos, H, E_idx, edge_attr, te_e_cont)
            atom_hids = atom_hids + self.node_blocks[layer_index](H)
            edge_hids = edge_hids + self.edge_blocks[layer_index](edge_attr)

        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]

        H = atom_hids
        edge_attr = self.bond_refine(batch, x, H, E_idx, edge_hids)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)

        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
        }
        if self.use_z == "pair_embedding":
            num_atoms = H.size(0)
            z_dense = torch.zeros(num_atoms, num_atoms, Z.size(-1), device=H.device)
            src, tgt = E_idx
            z_dense[src, tgt, :] = Z
            z_dense = 0.5 * (z_dense + z_dense.permute(1, 0, 2))
            Z = self.Z_lin(z_dense)
            out["Z_hat"] = Z
        return out


class DiTeBlockAli(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(
        self,
        hidden_size,
        edge_hidden_size,
        num_heads,
        mlp_expansion_ratio=4.0,
        use_z=None,
        n_vector_features=128,
        scale_dist_features=4,
        dropout=0.0,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.pre_attn_norm = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.pre_attn_norm_edge = BatchLayerNorm(edge_hidden_size, affine=False, eps=1e-6)
        #! AlphaFold3 has the scale as linear then sigmoid
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_edge_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(edge_hidden_size, 6 * edge_hidden_size, bias=True)
        )

        if scale_dist_features > 1:
            dist_size = n_vector_features * scale_dist_features
        else:
            dist_size = n_vector_features
        self.feature_embedder = MLP(hidden_size + hidden_size + edge_hidden_size + dist_size, hidden_size, hidden_size)

        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        # self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.d_head = hidden_size // num_heads
        self.use_z = use_z
        if use_z is not None:
            if use_z == "pair_embedding":
                self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, num_heads, bias=False))
            else:
                self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(dist_size, num_heads, bias=False))
            self.z_update = nn.Linear(hidden_size, hidden_size, bias=False)

        self.lin_edge0 = nn.Linear(hidden_size, edge_hidden_size, bias=False)
        self.lin_edge1 = nn.Linear(
            edge_hidden_size + n_vector_features * scale_dist_features, edge_hidden_size, bias=False
        )
        self.residual_ffn_norm = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.residual_ffn_norm_edge = BatchLayerNorm(edge_hidden_size, affine=False, eps=1e-6)
        self.ffn_edge = swiglu_ffn(edge_hidden_size, mlp_expansion_ratio, bias=False)

    def forward(
        self,
        batch: torch.Tensor,
        x: torch.Tensor,
        t_emb_h: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        t_emb_e: torch.Tensor = None,
        dist: torch.Tensor = None,
        Z: torch.Tensor = None,
        sbl=None,
    ):
        """
        This assume pytorch geometric batching so batch size of 1 so skip rotary as it depends on having an actual batch

        batch: N
        x: N x 256
        temb: N x 256
        edge_attr: E x 256
        edge_index: 2 x E
        """
        src, tgt = edge_index
        edge_batch = batch[src]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb_h).chunk(6, dim=1)
        (
            edge_shift_msa,
            edge_scale_msa,
            edge_gate_msa,
            edge_shift_mlp,
            edge_scale_mlp,
            edge_gate_mlp,
        ) = self.adaLN_edge_modulation(t_emb_e).chunk(6, dim=1)
        # Normalize x
        x_norm = modulate(self.pre_attn_norm(x, batch), shift_msa, scale_msa)
        edge_attr_norm = modulate(self.pre_attn_norm_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa)

        messages = self.feature_embedder(torch.cat([x_norm[src], x_norm[tgt], edge_attr_norm, dist], dim=-1))
        x_norm = scatter_mean(messages, src, dim=0)
        start = time.perf_counter()
        out = self.og_mha(x_norm, batch)
        torch.cuda.synchronize()
        mid = time.perf_counter()
        self.ali_mha(x_norm, batch, sbl)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"OG {mid - start} Ali {end - mid}")
        y = self.out_projection(out)

        # Gated Residual
        x = x + gate_msa * y
        # Feed Forward
        edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
        x = x + gate_mlp * self.ffn(modulate(self.residual_ffn_norm(x, batch), shift_mlp, scale_mlp))
        e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
        # import ipdb; ipdb.set_trace()
        edge_attr = edge + edge_gate_mlp * self.ffn_edge(
            modulate(self.residual_ffn_norm_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp)
        )
        if self.use_z == "pair_embedding":
            Z = Z + self.z_update(self.pyg_outer_update(batch, x))
            return x, edge_attr, Z
        else:
            return x, edge_attr, None

    def pyg_outer_update(self, batch, x):
        outer_products = []
        num_atoms = 0
        for molecule in range(max(batch.unique()) + 1):
            molecule_indices = (batch == molecule).nonzero(as_tuple=True)[0]
            molecule_features = x[molecule_indices]
            # Compute the outer product for the current molecule
            outer_product = molecule_features.unsqueeze(0) * molecule_features.unsqueeze(
                1
            )  # torch.einsum('ij,ik->ijk', molecule_features, molecule_features)
            outer_product = outer_product + outer_product.transpose(0, 1)
            outer_product = outer_product[~torch.eye(outer_product.size(0), dtype=torch.bool)].view(
                outer_product.size(0), -1
            )
            outer_products.append(outer_product.reshape(-1, molecule_features.shape[-1]))
            num_atoms += molecule_features.size(0)
        update = torch.cat(outer_products, dim=0)
        return update

    def ali_mha(self, x_norm, batch, sbl):
        # QKV projection
        qkv = self.qkv_proj(x_norm)  # seq_len, num_heads*head_dim
        new_id, mask, n_batch, max_b = sbl
        device = qkv.device
        # import ipdb; ipdb.set_trace()
        head_h = qkv.shape[-1]

        new_qkv = torch.zeros((n_batch * max_b, head_h), dtype=qkv.dtype, device=device)

        new_qkv[new_id, :] = qkv[:, :]

        new_qkv = new_qkv.reshape(n_batch, max_b, self.num_heads, -1).permute(
            (0, 2, 1, 3)
        )  # batch x num-head x max_atoms x D
        Q, K, V = new_qkv.chunk(3, dim=-1)

        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        # mask [1 1 num_atoms num_atoms] QKV = [1, num_heads, num_atoms, hidden//num_heads]

        attn_output = einops.rearrange(attn_output, "b h s d -> (b s) (h d)")[new_id, :]
        return attn_output

    def og_mha(self, x_norm, batch):
        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        # Q, K = self.norm_q(Q, batch), self.norm_k(K, batch)
        # Reshape Q, K, V to (1, seq_len, num_heads*head_dim)
        # if x.dim() == 2:
        Q = Q.unsqueeze(0)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)

        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=self.num_heads)
        # Reshape Q, K, V to (1, num_heads, seq_len, head_dim)
        Q, K, V = map(reshaper, (Q, K, V))

        attn_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(
            0
        )  #! if float it is added as the biasbut would still need a mask s -infs?

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask
        )  # mask [1 1 num_atoms num_atoms] QKV = [1, num_heads, num_atoms, hidden//num_heads]
        attn_output = einops.rearrange(attn_output, "b h s d -> b s (h d)").squeeze(0)
        return attn_output


class MoleculeDiffusionAli(nn.Module):
    def __init__(
        self,
        num_layers=10,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=22,
        edge_classes=5,
        num_heads=4,
        n_vector_features=128,
        scale_dist_features=4,
        use_z=None,
        attn_dropout=0.0,
        use_cross_product=False,
    ):
        super(MoleculeDiffusionAli, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)
        self.scale_dist_features = scale_dist_features

        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)

        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.use_z = use_z
        if use_z == "pair_embedding":
            self.Z_lin = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)

        for i in range(num_layers):
            self.dit_layers.append(
                DiTeBlockAli(
                    invariant_node_feat_dim,
                    invariant_edge_feat_dim,
                    num_heads,
                    mlp_expansion_ratio=4.0,
                    use_z=use_z,
                    n_vector_features=n_vector_features,
                    scale_dist_features=scale_dist_features,
                    dropout=attn_dropout,
                )
            )
            self.egnn_layers.append(
                EquivariantBlock(
                    invariant_node_feat_dim,
                    invariant_edge_feat_dim,
                    n_vector_features,
                    use_cross_product=use_cross_product,
                )
            )

        self.node_blocks = nn.ModuleList(
            [nn.Linear(invariant_node_feat_dim, invariant_node_feat_dim) for i in range(num_layers)]
        )
        self.edge_blocks = nn.ModuleList(
            [nn.Linear(invariant_edge_feat_dim, invariant_edge_feat_dim) for i in range(num_layers)]
        )

    def pyg_outer_update(self, batch, x):
        outer_products = []
        num_atoms = 0
        for molecule in range(max(batch.unique()) + 1):
            molecule_indices = (batch == molecule).nonzero(as_tuple=True)[0]
            molecule_features = x[molecule_indices]
            # Compute the outer product for the current molecule
            outer_product = molecule_features.unsqueeze(0) * molecule_features.unsqueeze(
                1
            )  # torch.einsum('ij,ik->ijk', molecule_features, molecule_features)
            outer_product = outer_product + outer_product.transpose(0, 1)
            outer_product = outer_product[~torch.eye(outer_product.size(0), dtype=torch.bool)].view(
                outer_product.size(0), -1
            )
            outer_products.append(outer_product.reshape(-1, molecule_features.shape[-1]))
            num_atoms += molecule_features.size(0)
        update = torch.cat(outer_products, dim=0)
        return update

    def shared_between_layers(self, batch, device="cuda"):
        particle_nums = batch.shape[0]
        _, batch_atoms = torch.unique_consecutive(batch, return_counts=True)
        n_batch = batch[-1].item() + 1

        max_b = torch.max(batch_atoms).item()

        cumsum_result = torch.cat(
            [torch.tensor([0], device=device, dtype=torch.long), torch.cumsum(max_b - batch_atoms, dim=0)[:-1]]
        )

        new_id = torch.repeat_interleave(cumsum_result, batch_atoms) + torch.arange(
            0, particle_nums, device=device, dtype=torch.long
        )

        mask = torch.tensor([[]], dtype=torch.bool, device=device)
        # max_b = batch.max().item()
        mask = torch.arange(max_b).cuda().unsqueeze(0).unsqueeze(0)  # [1, 1, max_b]
        mask = mask.expand(n_batch, max_b, max_b)  # [n_batch, max_b, max_b]
        mask = mask < batch_atoms.unsqueeze(1).unsqueeze(2)  # [n_batch, max_b, max_b]
        mask = mask & mask.transpose(1, 2)
        mask = mask.unsqueeze(1)

        return new_id, mask, n_batch, max_b

    def forward(self, batch, X, H, E_idx, E, t_discrete, t_continuous=None):
        torch.max(batch) + 1
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K
        H = self.atom_embedder(H)
        E = self.edge_embedder(E)
        te = self.time_embedding(t_discrete)
        te_h = te[batch]
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))
        te_e = te[edge_batch]
        if t_continuous:
            te_continuous = self.time_embedding(t_continuous)
            te_e_cont = te_continuous[edge_batch]
        else:
            te_e_cont = te_e
        edge_attr = E
        if self.use_z == "pair_embedding":
            Z = self.pyg_outer_update(batch, H)
        else:
            Z = None

        atom_hids = H
        edge_hids = edge_attr
        sbl = self.shared_between_layers(batch, pos.device)
        for layer_index in range(len(self.dit_layers)):
            # with nvtx.range(f"DiTe Layer {layer_index}"):
            distances = coord2distfn(pos, E_idx, self.scale_dist_features, batch)  # E x K
            H, edge_attr, Z = self.dit_layers[layer_index](batch, H, te_h, edge_attr, E_idx, te_e, distances, Z, sbl)
            # with nvtx.range(f"EGNN Layer {layer_index}"):
            pos = self.egnn_layers[layer_index](batch, pos, H, E_idx, edge_attr, te_e_cont)
            atom_hids = atom_hids + self.node_blocks[layer_index](H)
            edge_hids = edge_hids + self.edge_blocks[layer_index](edge_attr)

        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]

        H = atom_hids
        edge_attr = self.bond_refine(batch, x, H, E_idx, edge_hids)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)

        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
        }
        if self.use_z == "pair_embedding":
            num_atoms = H.size(0)
            z_dense = torch.zeros(num_atoms, num_atoms, Z.size(-1), device=H.device)
            src, tgt = E_idx
            z_dense[src, tgt, :] = Z
            z_dense = 0.5 * (z_dense + z_dense.permute(1, 0, 2))
            Z = self.Z_lin(z_dense)
            out["Z_hat"] = Z
        return out