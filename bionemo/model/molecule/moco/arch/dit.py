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

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
from torch_scatter import scatter, scatter_mean, scatter_softmax

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
    #! needs to be 0 CoM
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

    def forward(self, t, batch):
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


# def modulate(x, shift, scale):
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class DiTBlock(nn.Module):
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
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.norm1 = BatchLayerNorm(
            hidden_size, affine=False, eps=1e-6
        )  # LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = nn.MultiheadAttention(
        # embed_dim=hidden_size, num_heads=num_heads, bias=True, **block_kwargs
        # )
        self.norm2 = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        # mlp_hidden_dim = int(hidden_size * mlp_expansion_ratio)
        # self.ffn = swiglu_ln_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.ffn_norm = BatchLayerNorm(hidden_size)
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.norm_q = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.norm_k = BatchLayerNorm(hidden_size, affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.use_rotary = use_rotary
        if use_rotary:
            self.d_head = hidden_size // num_heads
            self.rotary = RotaryEmbedding(self.d_head)

        if use_z:
            self.use_z = use_z
            self.pair_bias = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 1, bias=False))
            self.mask_z = mask_z

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.num_heads, self.d_head))
        k = k.unflatten(-1, (self.num_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, batch: torch.Tensor, x: torch.Tensor, t_emb: torch.Tensor, Z: torch.Tensor = None):
        """
        This assume pytorch geometric batching so batch size of 1 so skip rotary as it depends on having an actual batch
        """
        if Z is not None:
            assert self.use_z

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Normalize x
        x_norm = modulate(self.norm1(x, batch), shift_msa, scale_msa)

        # QKV projection
        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        # Q, K = self.norm_q(Q), self.norm_k(K)
        Q, K = self.norm_q(Q, batch), self.norm_k(K, batch)
        # Reshape Q, K, V to (1, seq_len, num_heads*head_dim)
        if x.dim() == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)
            self.use_rotary = False

        if self.use_rotary:
            self._apply_rotary(Q, K)

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
        x = x + gate_mlp * self.ffn(
            self.ffn_norm(modulate(self.norm2(x, batch), shift_mlp, scale_mlp), batch)
        )  #! Using batch layer norm for PyG
        if Z is not None:
            Z = Z + x.unsqueeze(1) * x.unsqueeze(0)  # Z outer product
            if self.mask_z:
                Z = Z * mask.unsqueeze(-1)
            return x, Z
        return x


# Done: Create Attnention with norm and rotary, refer to https://github.com/evolutionaryscale/esm/blob/main/esm/layers/attention.py but they just use standard no bias
# Done add QK noramlization from ESM3 stemming from https://proceedings.mlr.press/v202/dehghani23a/dehghani23a.pdf see https://github.com/kyegomez/MegaVIT/blob/main/mega_vit/main.py#L84 and https://github.com/evolutionaryscale/esm/blob/main/esm/layers/attention.py#L48


# TODO: @Ali for EGNN should we remove the bias term from MLP's for X data?
class XEGNN(MessagePassing):
    """
    X only EGNN
    """

    def __init__(
        self,
        invariant_node_feat_dim=64,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")  #! This should be target to source
        self.message_input_size = 2 * invariant_node_feat_dim + 1  # + invariant_edge_feat_dim
        self.phi_message = MLP(self.message_input_size, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)
        self.coor_update_clamp_value = 10.0
        # self.reset_parameters()
        self.h_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.use_cross_product = True
        if self.use_cross_product:
            self.phi_x_cross = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1)
        self.x_norm = E3Norm()
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
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        if edge_attr is not None:
            edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feat = rel_dist
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

        H_out = H
        return X_out, H_out


class EdgeMixInLayer(MessagePassing):
    def __init__(
        self,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")  #! This should be target to source
        # self.pre_edge = MLP(invariant_edge_feat_dim, invariant_edge_feat_dim, invariant_edge_feat_dim)
        # self.edge_lin = MLP(2 * invariant_edge_feat_dim + 3, invariant_edge_feat_dim, invariant_edge_feat_dim)
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

    def forward(self, batch, X, H, edge_index, edge_attr):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)
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
        H_out = H + self.phi_h(torch.cat([H, m_i], dim=-1))  # self.h_norm(H, batch)
        return X_out, H_out


class EdgeMixOutLayer(MessagePassing):
    def __init__(
        self,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")  #! This should be target to source
        self.pre_edge = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_edge_feat_dim)
        # self.edge_lin = MLP(2 * invariant_edge_feat_dim + 3, invariant_edge_feat_dim, invariant_edge_feat_dim)
        self.message_input_size = invariant_node_feat_dim + invariant_node_feat_dim + 1
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

    def forward(self, batch, X, H, edge_index):
        X = X - scatter_mean(X, index=batch, dim=0, dim_size=X.shape[0])
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        edge_attr_feat = rel_dist  # torch.cat([edge_attr, rel_dist], dim=-1)
        m_ij = self.phi_message(torch.cat([H[target], H[source], edge_attr_feat], dim=-1))
        coor_wij = self.phi_x(m_ij)  # E x 1
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
        H_out = H + self.phi_h(torch.cat([H, m_i], dim=-1))

        edge_attr = self.pre_edge(m_ij)
        return X_out, H_out, edge_attr


class BondRefine(MessagePassing):
    def __init__(
        self,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")
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


class SE3MixAttention(nn.Module):
    def __init__(
        self,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=0,
    ):
        super(SE3MixAttention, self).__init__()
        self.equivariant_node_feature_dim = equivariant_node_feature_dim
        self.invariant_node_feat_dim = invariant_node_feat_dim
        self.invariant_edge_feat_dim = invariant_edge_feat_dim
        self.KV = MLP(
            invariant_node_feat_dim + 1, 2 * invariant_node_feat_dim, 2 * invariant_node_feat_dim, bias=False
        )
        self.Q = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_node_feat_dim, bias=False)
        # self.pair_bias = MLP(invariant_node_feat_dim, 4 * invariant_node_feat_dim, 1)
        # self.gate = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1, last_act='sigmoid')
        self.phi_h = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1, bias=False)
        # self.coor_update_clamp_value = 10.0
        self.x_norm = E3Norm()
        self.Q_norm = BatchLayerNorm(invariant_node_feat_dim)
        self.K_norm = BatchLayerNorm(invariant_node_feat_dim)

    def forward(self, batch, X, H, E_idx, ZE=None):
        # TODO is this the right direction for src and dst? we will remove this away from PyG but still should check this
        X = self.x_norm(X, batch)
        src, dst = E_idx
        Q = self.Q(H[dst])
        rel_coors = X[src] - X[dst]
        rel_dist = ((X[src] - X[dst]) ** 2).sum(dim=-1, keepdim=True)
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist + 1e-8))
        kv_input = torch.cat([rel_dist, H[src]], dim=-1)
        K, V = self.KV(kv_input).split(self.invariant_node_feat_dim, dim=-1)
        edge_batch, counts = torch.unique(batch, return_counts=True)
        edge_batch = torch.repeat_interleave(edge_batch, counts * (counts - 1))
        Q = self.Q_norm(Q, edge_batch)
        K = self.K_norm(K, edge_batch)
        # import ipdb; ipdb.set_trace()
        # B = self.pair_bias(ZE)
        attention_scores = (Q * K).sum(dim=-1) / (self.invariant_node_feat_dim**0.5)  # + B
        alpha_ij = scatter_softmax(attention_scores, dst, dim=0)  #! DO we want to gate maybe start simpler

        attention_output = alpha_ij.unsqueeze(-1) * V

        A = scatter(attention_output, dst, dim=0, dim_size=H.size(0), reduce='sum')
        # AX = scatter(attention_output * rel_dist, dst, dim=0, dim_size=H.size(0), reduce='sum')
        AH = scatter(attention_output * H[dst], dst, dim=0, dim_size=H.size(0), reduce='sum')

        X_out = X + scatter(
            X_rel_norm * self.phi_x(attention_output), index=dst, dim=0, reduce='sum', dim_size=X.shape[0]
        )
        H_out = H + self.phi_h(A * AH)
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
    temb = TimestepEmbedder(64)(time, batch_ligand)
    dit = DiTBlock(64, 8)
    out = dit(batch_ligand, H, temb[batch_ligand], Z)
    print("Success")
