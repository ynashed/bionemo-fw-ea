# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import functools
import math
import time as time_keeper

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
from torch_scatter import scatter, scatter_mean

from bionemo.model.molecule.moco.models.utils import PredictionHead


def on():
    return time_keeper.perf_counter()


def off(start, keyword=""):
    end = time_keeper.perf_counter()
    print(keyword, end - start)
    return end


def off_gpu(start, keyword=""):
    torch.cuda.synchronize()
    end = time_keeper.perf_counter()
    print(keyword, end - start)
    return end


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
        # import ipdb; ipdb.set_trace() #! the weight is what keeps it from being all identical
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


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def swiglu_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


class XEGNNK(nn.Module):
    """
    X only EGNN
    """

    def __init__(
        self,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
        n_vector_features=128,
        time_embedding_dim=256,
        input_edges=False,
        use_cross_product=True,
        prune_edges=False,
    ):
        super().__init__()  #! This should be target to source
        self.prune_edges = prune_edges
        self.message_input_size = (
            2 * invariant_node_feat_dim + n_vector_features + time_embedding_dim
        )  # + invariant_edge_feat_dim
        self.input_edges = input_edges
        if self.input_edges:
            self.message_input_size += invariant_edge_feat_dim
        self.phi_message = MLP(self.message_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, n_vector_features)
        self.coor_update_clamp_value = 10.0
        # self.reset_parameters()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.use_cross_product = use_cross_product
        if self.use_cross_product:
            self.phi_x_cross = MLP(invariant_node_feat_dim, invariant_node_feat_dim, n_vector_features)
        self.x_norm = E3Norm(n_vector_features)

    #     self.apply(self.init_)

    # # def reset_parameters(self):
    # def init_(self, module): #! this made it worse
    #     if type(module) in {nn.Linear}:
    #         # seems to be needed to keep the network from exploding to NaN with greater depths
    #         nn.init.xavier_normal_(module.weight)
    #         nn.init.zeros_(module.bias)
    def prune_edge_index(self, edge_index, rel_dist, batch, k=10):
        # Initialize a list to collect pruned edge indices and their corresponding distances
        pruned_edges = []
        pruned_distances = []

        # Number of nodes in the batch
        num_nodes = batch.size(0)

        # Iterate over each node to find the top-k closest nodes within the same batch
        for node in range(num_nodes):
            # Get the indices of edges where 'node' is the source (edge_index[0] == node)
            node_edges_mask = edge_index[0] == node
            node_edge_indices = torch.where(node_edges_mask)[0]

            # Get corresponding destination nodes and distances
            node_dists = rel_dist[node_edges_mask].sum(-1)

            # Sort by distance and select the top-k nearest neighbors
            _, topk_indices = torch.topk(node_dists, k, largest=False)

            # Select the top-k edges and distances
            topk_edge_indices = node_edge_indices[topk_indices]
            pruned_edges.append(edge_index[:, topk_edge_indices])
            pruned_distances.append(rel_dist[topk_indices])

        # Concatenate pruned edges and distances
        pruned_edges = torch.cat(pruned_edges, dim=1)
        pruned_distances = torch.cat(pruned_distances)

        return pruned_edges, pruned_distances

    def forward(self, batch, X, H, edge_index, edge_attr=None, te=None, sbl=None):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])[batch]
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
        # import ipdb; ipdb.set_trace()

        if self.prune_edges and not self.input_edges:
            test = scatter_mean(rel_dist.sum(-1), batch[source])
            edge_cut_mask = rel_dist.sum(-1) < test[batch[source]] / 2
            # edge_index.size(1)
            edge_index = edge_index[:, edge_cut_mask]
            # print(edge_index.size(1), " edges from", start_count)
            source, target = edge_index
            rel_coors = X[source] - X[target]
            rel_dist = (rel_coors.transpose(1, 2) ** 2).sum(dim=-1, keepdim=False)
            # edge_index, rel_dist = self.prune_edge_index(edge_index, rel_dist, batch)
            # source, target = edge_index
            # rel_coors = X[source] - X[target]
        if edge_attr is not None and self.input_edges:
            edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feat = rel_dist
        m_ij = self.phi_message(torch.cat([H[target], H[source], edge_attr_feat, te[batch[source]]], dim=-1))
        coor_wij = self.phi_x(m_ij)  # E x 3
        if self.coor_update_clamp_value:
            coor_wij.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
        # import ipdb; ipdb.set_trace()
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


class BondRefine(nn.Module):  #! can make this nn.Module to ensure no weird propagate error
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
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  # E
        edge_attr = self.edge_norm(edge_attr, edge_batch)
        infeats = torch.cat([H[target], H[source], rel_dist, edge_attr], dim=-1)
        return self.bond_norm(self.refine_layer(infeats), edge_batch)


class DiTeBlock(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(
        self,
        hidden_size,
        edge_feature_size,
        num_heads,
        mlp_expansion_ratio=4.0,
        n_vector_features=128,
        scale_dist_features=1,
        input_edges=False,
        output_edges=False,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.norm1 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)

        self.input_edges = input_edges
        self.output_edges = output_edges

        if scale_dist_features > 1:
            dist_size = n_vector_features * scale_dist_features
        else:
            dist_size = n_vector_features
        if self.input_edges:
            self.feature_embedder = MLP(2 * hidden_size + edge_feature_size + dist_size, hidden_size, hidden_size)
            self.adaLN_edge_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * edge_feature_size, bias=True)
            )
            self.norm1_edge = BatchLayerNorm(edge_feature_size, affine=False, eps=1e-6)
        else:
            self.feature_embedder = MLP(hidden_size + hidden_size + dist_size, hidden_size, hidden_size)
            self.adaLN_edge_modulation = None

        self.ffn_norm = BatchLayerNorm(hidden_size)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.d_head = hidden_size // num_heads

        if self.output_edges:
            if self.adaLN_edge_modulation is None:
                self.adaLN_edge_modulation = nn.Sequential(
                    nn.SiLU(), nn.Linear(hidden_size, 6 * edge_feature_size, bias=True)
                )
            self.lin_edge0 = nn.Linear(hidden_size, edge_feature_size, bias=False)
            # self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.lin_edge1 = nn.Linear(
                edge_feature_size + n_vector_features * scale_dist_features, edge_feature_size, bias=False
            )
            self.ffn_norm_edge = BatchLayerNorm(edge_feature_size)
            self.ffn_edge = swiglu_ffn(edge_feature_size, mlp_expansion_ratio, bias=False)
            self.norm2_edge = BatchLayerNorm(edge_feature_size, affine=False, eps=1e-6)
        # self.tanh = nn.GELU(approximate='tanh')

    def og_mha(self, x_norm, batch):
        # QKV projection
        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q, K = self.norm_q(Q, batch), self.norm_k(K, batch)
        # Reshape Q, K, V to (1, seq_len, num_heads*head_dim)
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

    def forward(
        self,
        batch: torch.Tensor,
        x: torch.Tensor,
        t_emb_h: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        t_emb_e: torch.Tensor = None,
        dist: torch.Tensor = None,
        sbl: torch.Tensor = None,
    ):
        """
        This assume pytorch geometric batching so batch size of 1 so skip rotary as it depends on having an actual batch

        batch: N
        x: N x 256
        temb: N x 256
        edge_attr: E x 256
        edge_index: 2 x E
        """

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb_h).chunk(6, dim=1)
        src, tgt = edge_index
        # Normalize x
        edge_batch = batch[src]
        x_norm = modulate(self.norm1(x, batch), shift_msa, scale_msa)
        # import ipdb; ipdb.set_trace()
        if self.input_edges:
            (
                edge_shift_msa,
                edge_scale_msa,
                edge_gate_msa,
                edge_shift_mlp,
                edge_scale_mlp,
                edge_gate_mlp,
            ) = self.adaLN_edge_modulation(t_emb_e).chunk(6, dim=1)

            edge_attr_norm = modulate(self.norm1_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa)
            messages = self.feature_embedder(torch.cat([x_norm[src], x_norm[tgt], edge_attr_norm, dist], dim=-1))
        else:
            messages = self.feature_embedder(torch.cat([x_norm[src], x_norm[tgt], dist], dim=-1))
        x_norm = scatter_mean(messages, src, dim=0)
        # attn_output = self.ali_mha(x_norm, batch, sbl) #!1.5 it/sec Causes NaN's everywhere after 37 iter #TODO need to add QK prenorm with mask?
        # if torch.isnan(attn_output).any():
        #     import ipdb; ipdb.set_trace()
        attn_output = self.og_mha(x_norm, batch)  #! No NaN 1.4 it/sec local bs 150
        y = self.out_projection(attn_output)

        # TODO: need to add in gate unsqueeze when we use batch dim
        # Gated Residual
        x = x + gate_msa * y
        # Feed Forward
        x = x + gate_mlp * self.ffn(self.ffn_norm(modulate(self.norm2(x, batch), shift_mlp, scale_mlp), batch))
        if self.output_edges:
            (
                _,
                _,
                edge_gate_msa,
                edge_shift_mlp,
                edge_scale_mlp,
                edge_gate_mlp,
            ) = self.adaLN_edge_modulation(
                t_emb_e
            ).chunk(6, dim=1)
            edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
            e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
            # import ipdb; ipdb.set_trace()
            edge_attr = edge + edge_gate_mlp * self.ffn_edge(
                self.ffn_norm_edge(
                    modulate(self.norm2_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp), edge_batch
                )
            )
        return x, edge_attr
        # return x, None


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


class MegalodonV2(nn.Module):
    def __init__(
        self,
        num_layers=10,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=16,
        edge_classes=5,
        num_heads=16,
        n_vector_features=128,
        prune_edges=False,
    ):
        super(MegalodonV2, self).__init__()
        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.coord_emb = nn.Linear(1, n_vector_features, bias=False)
        self.coord_pred = nn.Linear(n_vector_features, 1, bias=False)
        self.scale_dist_features = 4
        #! TODO do we need coord prediction head which is mlp then 0 CoM?
        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.attention_mix_layers = nn.ModuleList()
        for i in range(num_layers):
            self.dit_layers.append(
                DiTeBlock(
                    invariant_node_feat_dim,
                    invariant_edge_feat_dim,
                    num_heads=num_heads,
                    input_edges=i == 0,
                    output_edges=i == num_layers - 1,
                    scale_dist_features=self.scale_dist_features,
                    n_vector_features=n_vector_features,
                )
            )
            self.egnn_layers.append(
                XEGNNK(
                    invariant_node_feat_dim=invariant_node_feat_dim,
                    invariant_edge_feat_dim=invariant_edge_feat_dim,
                    n_vector_features=n_vector_features,
                    time_embedding_dim=invariant_node_feat_dim,
                    input_edges=i == num_layers - 1,
                    prune_edges=prune_edges,
                )
            )
        # self.h_feat_refine = DiTBlock(invariant_node_feat_dim, num_heads, use_z=False)
        self.node_blocks = nn.ModuleList(
            [nn.Linear(invariant_node_feat_dim, invariant_node_feat_dim) for i in range(num_layers)]
        )
        # self.edge_blocks = nn.ModuleList(
        #     [nn.Linear(invariant_edge_feat_dim, invariant_edge_feat_dim) for i in range(num_layers)]
        # )

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
        # mask = mask & mask.transpose(1, 2) #! causes nans in attention cannot mask out entire rows even if padding
        mask = mask.unsqueeze(1)
        # import ipdb; ipdb.set_trace()

        return new_id, mask, n_batch, max_b

    def forward(self, batch, X, H, E_idx, E, t):
        torch.max(batch) + 1
        # start = on()
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K
        # import ipdb; ipdb.set_trace()

        H = self.atom_embedder(H)
        E = self.edge_embedder(E)  # should be + n_vector_features not + 1
        te = self.time_embedding(t)
        # end = off_gpu(start, "inits")
        te_h = te[batch]
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  #
        te_e = te[edge_batch]
        edge_attr = E

        atom_hids = H
        # edge_hids = edge_attr
        # start = on()
        # sbl = self.shared_between_layers(batch, pos.device)
        sbl = None
        for layer_index in range(len(self.dit_layers)):
            # end = off_gpu(start, "shared between layers")
            distances = coord2distfn(pos, E_idx, self.scale_dist_features, batch)  # E x K
            # end = off_gpu(end, "coord2dist")
            # import ipdb; ipdb.set_trace()
            H, edge_attr = self.dit_layers[layer_index](batch, H, te_h, edge_attr, E_idx, te_e, distances, sbl)
            # end = off_gpu(end, "DiT")
            pos = self.egnn_layers[layer_index](batch, pos, H, E_idx, edge_attr, te, sbl)  #! TODO at time here
            # end = off_gpu(end, "EGNN")
            atom_hids = atom_hids + self.node_blocks[layer_index](H)
            # start = off_gpu(end, "residual")
            # edge_hids = edge_hids + self.edge_blocks[layer_index](edge_attr)

        X = self.coord_pred(pos).squeeze(-1)
        x = X - scatter_mean(X, index=batch, dim=0)[batch]
        H = atom_hids
        edge_attr = self.bond_refine(batch, X, H, E_idx, edge_attr)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)
        # end = off_gpu(start, "output layer")
        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
            # "Z_hat": z_logits,
        }
        return out
