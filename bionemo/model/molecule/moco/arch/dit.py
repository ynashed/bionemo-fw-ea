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
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import LayerNorm as BatchLayerNorm
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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class DiTBlock(nn.Module):
    """
    Mimics DiT block
    """

    def __init__(self, hidden_size, num_heads, mlp_expansion_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, bias=True, **block_kwargs
        )  # Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_expansion_ratio)
        self.mlp = swiglu_ln_ffn(
            hidden_size, mlp_hidden_dim, bias=False
        )  # MLP(input_dim=hidden_size, hidden_size=mlp_hidden_dim, output_dim=hidden_size, num_hidden_layers=0, dropout=0, activation="gelu_tanh")
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, t_emb, z=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# TODO: Create Attnention with norm and rotary, refer to https://github.com/evolutionaryscale/esm/blob/main/esm/layers/attention.py but they just use standard no bias
# TODOL add QK noramlization from ESM3 stemming from https://proceedings.mlr.press/v202/dehghani23a/dehghani23a.pdf see https://github.com/kyegomez/MegaVIT/blob/main/mega_vit/main.py#L84 and https://github.com/evolutionaryscale/esm/blob/main/esm/layers/attention.py#L48


# TODO: @Ali for EGNN should we remove the bias term from MLP's for X data?
class XEGNN(MessagePassing):
    """
    X only EGNN
    """

    def __init__(
        self,
        # equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=0,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")  #! This should be target to source
        self.message_input_size = invariant_node_feat_dim + invariant_node_feat_dim + 1 + invariant_edge_feat_dim
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
        X = self.x_norm(X, batch)
        H = self.h_norm(H, batch)
        source, target = edge_index
        rel_coors = X[source] - X[target]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)
        if edge_attr:
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
        return X_out, H_out, edge_attr


class SE3MixAttention(nn.Module):
    def __init__(
        self,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=0,
    ):
        super.__init__()
        self.equivariant_node_feature_dim = equivariant_node_feature_dim
        self.invariant_node_feat_dim = invariant_node_feat_dim
        self.invariant_edge_feat_dim = invariant_edge_feat_dim
        self.KV = MLP(
            invariant_node_feat_dim + 1, 2 * invariant_node_feat_dim, 2 * invariant_node_feat_dim, bias=False
        )
        self.Q = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_node_feat_dim, bias=False)
        self.pair_bias = MLP(invariant_node_feat_dim, 4 * invariant_node_feat_dim, 1)
        self.gate = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1, last_act='sigmoid')
        self.phi_h = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_x = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1, bias=False)
        self.coor_update_clamp_value = 10.0
        self.x_norm = E3Norm()
        self.Q_norm = nn.LayerNorm(invariant_node_feat_dim)
        self.K_norm = nn.LayerNorm(invariant_node_feat_dim)

    def forward(self, batch, X, H, E_idx, ZE):
        X = self.x_norm(X, batch)
        src, dst = E_idx
        Q = self.Q(H[dst])
        rel_coors = X[src] - X[dst]
        rel_dist = ((X[src] - X[dst]) ** 2).sum(dim=-1, keepdim=True)
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist + 1e-8))
        kv_input = torch.cat([rel_dist, H[src]], dim=-1)
        K, V = self.KV(kv_input).split(self.invariant_node_feat_dim, dim=-1)

        Q = self.Q_norm(Q)
        K = self.K_norm(K)

        B = self.pair_bias(ZE)
        attention_scores = (Q * K).sum(dim=-1) / (self.invariant_node_feat_dim**0.5) + B
        alpha_ij = self.gate(H)[src, dst] * attention_scores

        attention_output = alpha_ij.unsqueeze(-1) * V  # [src]  # Shape: [num_edges, num_heads, head_dim]

        A = scatter(attention_output, dst, dim=0, dim_size=H.size(0), reduce='sum')
        AX = scatter(attention_output * X_rel_norm, dst, dim=0, dim_size=H.size(0), reduce='sum')
        AH = scatter(attention_output * H[dst], dst, dim=0, dim_size=H.size(0), reduce='sum')

        X_out = X + self.phi_x(A * AX)
        H_out = H + self.phi_h(A * AH)
        return X_out, H_out
