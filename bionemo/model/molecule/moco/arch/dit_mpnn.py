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
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_mean

from bionemo.model.molecule.moco.arch.dite import modulate, swiglu_ffn
from bionemo.model.molecule.moco.arch.rotary import RotaryEmbedding


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
    def __init__(self, n_vector_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((1, 1, n_vector_features)))  # Separate weights for each channel
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        # pos is expected to be of shape [n, 3, n_vector_features]
        # import ipdb; ipdb.set_trace()
        norm = torch.norm(pos, dim=1, keepdim=True)  # Normalize over the 3 dimension
        batch_size = int(batch.max()) + 1
        mean_norm = scatter_mean(norm, batch, dim=0, dim_size=batch_size)
        new_pos = self.weight * pos / (mean_norm[batch] + self.eps)
        return new_pos


class DiTeMPNN(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_expansion_ratio=4.0,
        use_z=False,
        mask_z=True,
        use_rotary=False,
        n_vector_features=128,
        dropout=0.0,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.edge_emb = nn.Linear(hidden_size + n_vector_features, hidden_size)
        self.norm1 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)

        self.norm1_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)

        self.norm2_node = BatchLayerNorm(hidden_size)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_edge_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.use_rotary = use_rotary
        self.d_head = hidden_size // num_heads
        if use_rotary:
            # self.d_head = hidden_size // num_heads
            self.rotary = RotaryEmbedding(self.d_head)
        self.use_z = use_z
        if use_z:
            self.use_z = use_z
            self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 1, bias=False))
            self.mask_z = mask_z

        self.node2edge_lin = nn.Linear(hidden_size, hidden_size)
        self.lin_edge0 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.ffn_norm_edge = BatchLayerNorm(hidden_size)
        self.ffn_edge = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.tanh = nn.GELU(approximate='tanh')

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.num_heads, self.d_head))
        k = k.unflatten(-1, (self.num_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

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
        if Z is not None:
            assert self.use_z
        src, tgt = edge_index
        h_in_node = x
        h_in_edge = edge_attr
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
        edge_attr = self.edge_emb(torch.cat([edge_attr, dist], dim=-1))
        # Normalize x
        x = modulate(self.norm1(x, batch), shift_msa, scale_msa)
        prev_edge_attr = edge_attr
        edge_attr = modulate(self.norm1_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa)

        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.reshape(-1, self.num_heads, self.d_head)
        K = K.reshape(-1, self.num_heads, self.d_head)
        V = V.reshape(-1, self.num_heads, self.d_head)

        # Gather the query, key, and value tensors for the source and target nodes
        query_i = Q[src]  # [E, heads, d_head]
        key_j = K[tgt]  # [E, heads, d_head]
        value_j = V[tgt]  # [E, heads, d_head]

        # Message function computes attention coefficients and messages for edges
        # Edge attributes are transformed and reshaped to match the head and channel dimensions
        edge_attn = self.lin_edge0(edge_attr).view(-1, self.num_heads, self.d_head)  # [E, heads, d_head]
        edge_attn = self.tanh(edge_attn)  # [E, heads, d_head]

        # Compute the attention scores using dot-product attention mechanism
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.d_head)  # [E, heads]

        # Apply softmax to normalize attention scores across all edges directed to the same node
        alpha = softmax(alpha, tgt)  # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Apply dropout to attention scores

        # Multiply normalized attention scores with the value tensor to compute the messages
        # import ipdb; ipdb.set_trace()
        msg = value_j
        msg = msg * self.lin_edge1(edge_attr).view(-1, self.num_heads, self.d_head)
        msg = msg * alpha.view(-1, self.num_heads, 1)  # [E, heads, d_head]

        # Aggregate messages to the destination nodes
        out = scatter(msg, tgt, dim=0, reduce='sum', dim_size=x.size(0))  # [N, heads, d_head]

        # Merge the heads and the output channels
        out = out.view(
            -1, self.num_heads * self.d_head
        )  # [N, heads * d_head] #? no linear layer for the first h_node? to aggregate the MHA?

        h_node = out
        h_edge = h_node[src] + h_node[tgt]
        h_edge = self.node2edge_lin(h_edge)

        h_node = h_in_node + gate_msa * h_node
        h_node = modulate(self.norm2_node(h_node, batch), shift_mlp, scale_mlp)
        h_out = h_node + gate_mlp * self.ffn(h_node)

        h_edge = h_in_edge + edge_gate_msa * h_edge
        h_edge = modulate(self.norm2_edge(h_edge, edge_batch), edge_shift_mlp, edge_scale_mlp)
        h_edge_out = prev_edge_attr + h_edge + edge_gate_mlp * self.ffn_edge(h_edge)

        return h_out, h_edge_out
