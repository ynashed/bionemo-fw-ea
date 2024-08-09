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

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
from torch_scatter import scatter_mean

from bionemo.model.molecule.moco.arch.dit import (
    MLP,
    BondRefine,
)
from bionemo.model.molecule.moco.arch.equiformer.equiformer import equivariant_transformer_l2
from bionemo.model.molecule.moco.arch.scratch.mpnn import TimestepEmbedder
from bionemo.model.molecule.moco.models.utils import PredictionHead


def coord2dist(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    return radial


def coord2distfn(x, edge_index, scale_dist_features=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    # import ipdb; ipdb.set_trace()
    if scale_dist_features >= 2:
        # dotproduct = x[row] * x[col]  # shape [num_edges, 3, k]
        # dotproduct = dotproduct.sum(-2)  # sum over the spatial dimension, shape [num_edges, k]
        # # Concatenate the computed features
        # radial = torch.cat([radial, dotproduct], dim=-1)  # shape [num_edges, 2*k]
        dotproduct = (x[row] * x[col]).sum(dim=-1, keepdim=True)  # shape [num_edges, 1]
        radial = torch.cat([radial, dotproduct], dim=-1)  # shape [num_edges, 2]
    if scale_dist_features == 4:
        p_i, p_j = x[edge_index[0]], x[edge_index[1]]
        d_i, d_j = (
            torch.pow(p_i, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
            torch.pow(p_j, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
        )
        radial = torch.cat([radial, d_i, d_j], dim=-1)

    return radial


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


class DiTeBlock(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_expansion_ratio=4.0,
        use_z=True,
        mask_z=True,
        use_rotary=False,
        n_vector_features=128,
        scale_dist_features=1,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.norm1 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        if scale_dist_features > 1:
            dist_size = n_vector_features * scale_dist_features
        else:
            dist_size = n_vector_features
        self.feature_embedder = MLP(hidden_size + hidden_size + hidden_size + dist_size, hidden_size, hidden_size)
        self.norm1_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm2_edge = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)

        self.ffn_norm = BatchLayerNorm(hidden_size)
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

        if use_z:
            self.use_z = use_z
            self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 1, bias=False))
            self.mask_z = mask_z

        self.lin_edge0 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.lin_edge1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_edge1 = nn.Linear(hidden_size + n_vector_features * scale_dist_features, hidden_size, bias=False)
        self.ffn_norm_edge = BatchLayerNorm(hidden_size)
        self.ffn_edge = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        # self.tanh = nn.GELU(approximate='tanh')

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
        edge_batch: torch.Tensor = None,
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

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb_h).chunk(6, dim=1)
        (
            edge_shift_msa,
            edge_scale_msa,
            edge_gate_msa,
            edge_shift_mlp,
            edge_scale_mlp,
            edge_gate_mlp,
        ) = self.adaLN_edge_modulation(t_emb_e).chunk(6, dim=1)
        src, tgt = edge_index
        # Normalize x
        x_norm = modulate(self.norm1(x, batch), shift_msa, scale_msa)

        edge_attr_norm = modulate(self.norm1_edge(edge_attr, edge_batch), edge_shift_msa, edge_scale_msa)
        messages = self.feature_embedder(torch.cat([x_norm[src], x_norm[tgt], edge_attr_norm, dist], dim=-1))
        x_norm = scatter_mean(messages, src, dim=0)

        # QKV projection
        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q, K = self.norm_q(Q, batch), self.norm_k(K, batch)
        # Reshape Q, K, V to (1, seq_len, num_heads*head_dim)
        if x.dim() == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)
            self.use_rotary = False

        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=self.num_heads)
        # Reshape Q, K, V to (1, num_heads, seq_len, head_dim)
        Q, K, V = map(reshaper, (Q, K, V))

        if x.dim() == 2:
            attn_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                0
            )  #! if float it is added as the biasbut would still need a mask s -infs?
        else:
            attn_mask = batch

        if Z is not None:
            if x.dim() == 2:
                mask = torch.ones((x.size(0), x.size(0)))
                if self.mask_z:
                    mask.fill_diagonal_(0)
                attn_mask = attn_mask.float()
                attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
                attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
                bias = (self.pair_bias(Z).squeeze(-1) * mask).unsqueeze(0).unsqueeze(0)
                attn_mask += bias
            else:
                raise ValueError("Have not implemented Batch wise pair embedding update")

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask
        )  # mask [1 1 num_atoms num_atoms] QKV = [1, num_heads, num_atoms, hidden//num_heads]
        attn_output = einops.rearrange(attn_output, "b h s d -> b s (h d)").squeeze(0)
        y = self.out_projection(attn_output)

        # TODO: need to add in gate unsqueeze when we use batch dim
        # Gated Residual
        x = x + gate_msa * y
        # Feed Forward
        edge = edge_attr + edge_gate_msa * self.lin_edge0((y[src] + y[tgt]))
        x = x + gate_mlp * self.ffn(self.ffn_norm(modulate(self.norm2(x, batch), shift_mlp, scale_mlp), batch))
        e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
        # import ipdb; ipdb.set_trace()
        edge_attr = edge + edge_gate_mlp * self.ffn_edge(
            self.ffn_norm_edge(modulate(self.norm2_edge(e_in, edge_batch), edge_shift_mlp, edge_scale_mlp), edge_batch)
        )
        return x, edge_attr


class MoleculeDiTEquiformer(nn.Module):
    def __init__(
        self,
        num_layers=8,
        equiformer_sub_layers=2,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=16,
        edge_classes=5,
        num_heads=4,
        n_vector_features=1,
        basis_type='edge_attr',
    ):
        super(MoleculeDiTEquiformer, self).__init__()
        assert n_vector_features == 1

        self.atom_embedder = MLP(atom_classes, invariant_node_feat_dim, invariant_node_feat_dim)
        self.edge_embedder = MLP(edge_classes, invariant_edge_feat_dim, invariant_edge_feat_dim)

        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features

        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.equiformer_layers = nn.ModuleList()
        # self.attention_mix_layers = nn.ModuleList()
        # hurts equivariance tolerance
        # self.h_norms = nn.ModuleList()
        # self.x_norms = nn.ModuleList()
        for i in range(num_layers):
            self.dit_layers.append(
                DiTeBlock(
                    invariant_node_feat_dim,
                    num_heads,
                    use_z=False,
                    n_vector_features=n_vector_features,
                    scale_dist_features=4,
                )
            )
            # self.h_norms.append(BatchLayerNorm(invariant_node_feat_dim))
            # self.x_norms.append(E3Norm(n_vector_features))
            self.equiformer_layers.append(
                equivariant_transformer_l2(
                    invariant_node_feat_dim,
                    radius=20,
                    basis_type=basis_type,
                    num_basis=64,
                    num_layers=equiformer_sub_layers,
                )
            )

        # self.h_feat_refine = DiTBlock(invariant_node_feat_dim, num_heads, use_z=False)

    def forward(self, batch, X, H, E_idx, E, t):
        torch.max(batch) + 1
        # pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K
        pos = X

        H = self.atom_embedder(H)
        E = self.edge_embedder(E)  # should be + n_vector_features not + 1
        te = self.time_embedding(t)
        te_h = te[batch]
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))  #
        te_e = te[edge_batch]

        edge_attr = E
        # import ipdb; ipdb.set_trace()
        for layer_index in range(len(self.dit_layers)):
            # distances = coord2dist(pos, E_idx) # E x K
            distances = coord2distfn(pos, E_idx, scale_dist_features=4)
            H, edge_attr = self.dit_layers[layer_index](batch, H, te_h, edge_attr, E_idx, te_e, distances, edge_batch)

            X_eg, H = self.equiformer_layers[layer_index](pos, H, E_idx, edge_attr, te_e)  # ! TODO at time here

            pos = X_eg - scatter_mean(X_eg, index=batch, dim=0, dim_size=X.shape[0])[batch]

        # X = self.coord_pred(pos).squeeze(-1)
        x = pos  # X - scatter_mean(X, index=batch, dim=0)[batch]

        # H = self.h_feat_refine(batch, H, te_h)
        edge_attr = self.bond_refine(batch, x, H, E_idx, edge_attr)

        h_logits, _ = self.atom_type_head(batch, H)
        e_logits, _ = self.edge_type_head.predict_edges(batch, edge_attr, E_idx)

        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
        }

        return out
