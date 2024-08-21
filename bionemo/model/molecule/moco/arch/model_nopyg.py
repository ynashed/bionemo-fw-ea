import functools
import math
from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm

from bionemo.model.molecule.moco.arch.egnn import LayerNormNoBias, Linear


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


def calc_com(coords, node_mask=None):
    """Calculates the centre of mass of a pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM of pointclouds with imaginary nodes excluded, shape [*, 1, 3]
    """

    node_mask = torch.ones_like(coords[..., 0]) if node_mask is None else node_mask

    assert node_mask.shape == coords[..., 0].shape

    num_nodes = node_mask.sum(dim=-1)
    real_coords = coords * node_mask.unsqueeze(-1)
    com = real_coords.sum(dim=-2) / num_nodes.unsqueeze(-1)
    return com.unsqueeze(-2)


def zero_com(coords, node_mask=None):
    """Sets the centre of mass for a batch of pointclouds to zero for each pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM-free coordinates, where imaginary nodes are excluded from CoM calculation
    """

    com = calc_com(coords, node_mask=node_mask)
    shifted = coords - com
    return shifted


class PredictionHead(nn.Module):
    def __init__(self, num_classes, feat_dim, discrete=True, edge_prediction=False, distance_prediction=False):
        super().__init__()
        self.num_classes = num_classes
        self.discrete = discrete
        self.projection = torch.nn.Sequential(
            Linear(feat_dim, feat_dim),
            nn.SiLU(),
            Linear(feat_dim, num_classes),
        )
        if edge_prediction:
            self.post_process = torch.nn.Sequential(
                Linear(num_classes, num_classes),
                nn.SiLU(),
                Linear(num_classes, num_classes),
            )
        # if distance_prediction:
        #     self.post_process = MLP(feat_dim, feat_dim, num_classes, last_act="sigmoid")
        #     self.embedding = MLP(feat_dim, feat_dim, feat_dim)

    #! Even if we have a masking state we still predict it but always mask it even on the forward pass as done in MultiFlow. The loss is taken over teh logits so its not masked
    def forward(self, mask, H):
        logits = self.projection(H) * mask.unsqueeze(-1)
        if self.discrete:
            probs = F.softmax(logits, dim=-1)  # scatter_softmax(logits, index=batch, dim=0, dim_size=H.size(0))
        else:
            probs = zero_com(logits, mask)
        return logits, probs

    def predict_edges(self, mask, E):
        #! EQGAT also uses hi hj and Xi-Xj along with f(E) see coordsatomsbonds.py line 121 https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/e3moldiffusion/coordsatomsbonds.py#L121C32-L121C44
        # import ipdb; ipdb.set_trace()
        e_dense = self.projection(E)
        e = 0.5 * (e_dense + e_dense.permute(0, 2, 1, 3))
        logits = self.post_process(e) * mask.unsqueeze(-1)  # E x 5
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    # def predict_distances(self, batch, Z):
    #     input = Z + Z.permute(1, 0, 2)
    #     logits = self.post_process(input) * self.embedding(input)
    #     logits = self.projection(logits).squeeze(-1)
    #     return logits


class E3Norm(nn.Module):
    """
    E3Normalization based on mean of norm of positions

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_vector_features: int = 1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        if n_vector_features > 1:
            self.weight = nn.Parameter(2 * torch.ones((1, 1, n_vector_features)))
        else:
            self.weight = nn.Parameter(2 * torch.ones((1, 1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    # [b, a, 3]  # [b, a]
    def forward(self, pos: torch.Tensor, mask: torch.Tensor = None):
        # import ipdb; ipdb.set_trace()
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]

        norm = torch.norm(pos * mask, dim=2, keepdim=False)  # [B N K]
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=False)

        new_pos = self.weight * pos / (mean_norm.unsqueeze(1) + self.eps)

        return new_pos * mask


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


def modulate(x, shift, scale, unsqueeze=True):
    if unsqueeze:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
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


class BondRefine(nn.Module):  #! can make this nn.Module to ensure no weird propagate error
    def __init__(
        self,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
    ):
        super().__init__()
        self.h_norm = LayerNorm(invariant_node_feat_dim)
        self.edge_norm = LayerNorm(invariant_edge_feat_dim)
        self.bond_norm = LayerNorm(invariant_edge_feat_dim)
        in_feats = 2 * invariant_node_feat_dim + 1 + invariant_edge_feat_dim
        self.refine_layer = torch.nn.Sequential(
            Linear(in_feats, invariant_edge_feat_dim),
            nn.SiLU(),
            Linear(invariant_edge_feat_dim, invariant_edge_feat_dim),
        )

    def forward(self, mask, X, H, edge_attr, edge_mask):
        # import ipdb; ipdb.set_trace()
        n_atoms = X.size(1)
        H = self.h_norm(H * mask.unsqueeze(-1))
        rel_vec = X.unsqueeze(-2) - X.unsqueeze(-3)  # [b, a, a, 3]
        rel_dist = (rel_vec**2).sum(dim=-1, keepdim=True)  # [b, a, a, 1]
        edge_attr = self.edge_norm(edge_attr * edge_mask.unsqueeze(-1))
        h1 = H[:, :, None, :].repeat(1, 1, n_atoms, 1)
        h2 = H[:, None, :, :].repeat(1, n_atoms, 1, 1)
        h_message = torch.cat([h1, h2, edge_attr, rel_dist], dim=-1)
        return self.bond_norm(self.refine_layer(h_message) * edge_mask.unsqueeze(-1))


class EGNNBlock(nn.Module):
    """
    X and H EGNN
    """

    def __init__(
        self,
        n_vector_features: int = 1,
        invariant_node_feat_dim: int = 64,
        invariant_edge_feat_dim: int = 0,
        time_embedding_dim: int = 0,
        activation: nn.Module = nn.SiLU(),
        use_clamp: bool = True,
        coords_range: float = 10.0,
        norm_constant: float = 1,
        aggregation_method: Literal['sum'] = 'sum',
        use_cross_product: bool = False,
        attenion_type: Literal["None", "Simple"] = "None",
        update_nodes: bool = False,
        input_edges: bool = False,
    ):
        super(EGNNBlock, self).__init__()

        self.use_clamp = use_clamp
        self.coors_range = coords_range

        self.invariant_node_feat_dim = invariant_node_feat_dim

        self.message_in_dim = 2 * invariant_node_feat_dim + n_vector_features + time_embedding_dim

        if input_edges:
            self.message_in_dim += invariant_edge_feat_dim

        self.h_norm = LayerNormNoBias(normalized_shape=(invariant_node_feat_dim), elementwise_affine=True)

        self.x_norm = E3Norm(n_vector_features=n_vector_features)

        self.phi_x = nn.Sequential(
            *[
                Linear(self.invariant_node_feat_dim, self.invariant_node_feat_dim),
                activation,
                Linear(self.invariant_node_feat_dim, n_vector_features),
            ]
        )

        self.phi_e = nn.Sequential(
            *[
                Linear(self.message_in_dim, self.message_in_dim),
                activation,
                Linear(self.message_in_dim, invariant_node_feat_dim),
            ]
        )
        self.update_nodes = update_nodes
        if self.update_nodes:
            self.phi_h = nn.Sequential(
                *[
                    Linear(2 * self.invariant_node_feat_dim, self.invariant_node_feat_dim),
                    activation,
                    Linear(self.invariant_node_feat_dim, invariant_node_feat_dim),
                ]
            )

        self.attenion_type = attenion_type

        if self.attenion_type == "Simple":
            self.phi_att = nn.Sequential(
                *[
                    Linear(self.invariant_node_feat_dim, self.invariant_node_feat_dim),
                    activation,
                    Linear(self.invariant_node_feat_dim, 1),
                    nn.Sigmoid(),
                ]
            )
        # else:
        #     self.phi_att = TaylorSeriesLinearAttn(
        #         dim=self.invariant_node_feat_dim,
        #         dim_head=max(4, self.invariant_node_feat_dim // 2),
        #         heads=1,
        #         one_headed_kv=1,
        #         prenorm=True,
        #     )

        self.use_cross_product = use_cross_product
        if self.use_cross_product:
            self.phi_x_cross = nn.Sequential(
                *[
                    Linear(self.invariant_node_feat_dim, self.invariant_node_feat_dim),
                    activation,
                    Linear(self.invariant_node_feat_dim, n_vector_features),
                ]
            )

        self.norm_constant = norm_constant
        self.aggregation_method = aggregation_method

        if use_clamp:
            self.clamp = self._clamp
        else:
            self.clamp = nn.Identity()

    def _clamp(self, x):
        return torch.clamp(x, min=-self.coors_range, max=self.coors_range)

    def forward(
        self, mask: Tensor, x: Tensor, h: Tensor, edge_mask: Tensor, edge_attr: Tensor = None, t: Tensor = None
    ):
        """_summary_

        Args:
            x (Tensor): [b, a, 3, K]
            h (Tensor): [b, a, d_h]
            mask (Tensor): [b, a]
            t (Tensor): [b, d_t]
            edge_attr (Optiona[Tensor]): [b, a, a, d_e]

        Returns:
            Updated x position
        """
        bs, n_atoms, _ = h.shape
        x = self.x_norm(x, mask)
        h_norm = self.h_norm(h * mask.unsqueeze(-1))

        rel_vec = x.unsqueeze(-4) - x.unsqueeze(-3)  # [b, a, a, 3, K]
        rel_dist = (rel_vec**2).sum(dim=3, keepdim=False)  # [b, a, a, K]

        if edge_attr is not None:
            edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)  # [b, a, a, K + d_e]
        else:
            edge_attr_feat = rel_dist

        h1 = h_norm[:, :, None, :].repeat(1, 1, n_atoms, 1)
        h2 = h_norm[:, None, :, :].repeat(1, n_atoms, 1, 1)

        h_message = torch.cat([h1, h2, edge_attr_feat], dim=-1)  # b, a, a,  d_m
        if t is not None:
            h_message = torch.cat([h_message, t.view(bs, 1, 1, -1).repeat(1, n_atoms, n_atoms, 1)], dim=-1)

        # mask = mask.int()
        # mask = (mask[:, :, None] + mask[:, None, :] > 1).float()

        # mask = mask.unsqueeze(-1)

        m_ij = self.phi_e(h_message * edge_mask.unsqueeze(-1))  # b, a, a,  d_m

        # if self.attenion_type == "Simple":
        #     att_val = self.phi_att(m_ij)  # b, a, a,  1
        #     m_ij = m_ij * att_val * mask  # b, a, a,  d_m
        # m_ij = torch.sum(m_ij * mask, dim=-2)  # b, a, d_h

        if self.update_nodes:
            h_out = (h_norm + self.phi_h(torch.cat([h, m_ij], dim=-1))) * mask.unsqueeze(-1)
        else:
            h_out = h

        x_updates_vals = self.clamp(self.phi_x(m_ij) * edge_mask.unsqueeze(-1)).unsqueeze(-2)  # b, a, a, K
        # x_updates_vals = torch.sum(x_updates_vals, dim=-2)  # b, a, k

        if self.use_cross_product:
            x_updates_cross = self.clamp(self.phi_x_cross(m_ij) * edge_mask.unsqueeze(-1)).unsqueeze(-2)

        if self.aggregation_method == 'sum':
            x_update = torch.sum(
                x_updates_vals * rel_vec / (self.norm_constant + torch.sqrt(rel_dist.unsqueeze(-2) + 1e-8)), dim=2
            )  # b, a, 3
            if self.use_cross_product:
                rel_cross = torch.linalg.cross(x.unsqueeze(-3), x.unsqueeze(-4), dim=-2)
                x_update = x_update + (
                    (rel_cross) / (self.norm_constant + rel_cross.norm(dim=-2, keepdim=True)) * x_updates_cross
                ).sum(dim=2)

        x = x + x_update

        return x, h_out


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
        self.edge_feature_size = edge_feature_size
        self.num_heads = num_heads
        self.input_adaln_norm = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        if scale_dist_features > 1:
            dist_size = n_vector_features * scale_dist_features
        else:
            dist_size = n_vector_features
        # self.feature_embedder = MLP(hidden_size + hidden_size + hidden_size + dist_size, hidden_size, hidden_size)
        # import ipdb; ipdb.set_trace()
        self.message_in_dim = 2 * hidden_size + dist_size
        self.input_edges = input_edges

        if self.input_edges:
            self.message_in_dim += edge_feature_size
            self.adaLN_edge_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * edge_feature_size, bias=True)
            )
            self.input_adaln_norm_edge = LayerNorm(edge_feature_size, elementwise_affine=False, eps=1e-6)

        self.feature_embedder = nn.Sequential(
            *[
                Linear(self.message_in_dim, hidden_size),
                nn.SiLU(),
                Linear(hidden_size, hidden_size),
            ]
        )
        # Single linear layer for QKV projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.norm_q = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_k = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        # Feed Forward
        self.d_head = hidden_size // num_heads
        self.ffn = swiglu_ffn(hidden_size, mlp_expansion_ratio, bias=False)
        self.ff_norm_inner = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output_edges = output_edges
        if self.output_edges:
            self.adaLN_edge_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * edge_feature_size, bias=True)
            )
            self.lin_edge0 = nn.Linear(hidden_size, edge_feature_size, bias=False)
            self.lin_edge1 = nn.Linear(
                edge_feature_size + n_vector_features * scale_dist_features, edge_feature_size, bias=False
            )
            self.ff_norm_inner_edge = LayerNorm(edge_feature_size, elementwise_affine=False, eps=1e-6)
            self.ffn_edge = swiglu_ffn(edge_feature_size, mlp_expansion_ratio, bias=False)

    def forward(
        self,
        mask: torch.Tensor,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_mask: torch.Tensor = None,
        dist: torch.Tensor = None,
        Z: torch.Tensor = None,
    ):
        """
        This assume pytorch geometric batching so batch size of 1 so skip rotary as it depends on having an actual batch

        batch: N
        x: B x N x 3 x K
        temb: B x N x 256
        edge_attr: E x 256
        edge_index: 2 x E
        """
        n_atoms = x.size(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        # import ipdb; ipdb.set_trace()
        x_norm = modulate(self.input_adaln_norm(x * mask.unsqueeze(-1)), shift_msa, scale_msa)
        h1 = x_norm[:, :, None, :].repeat(1, 1, n_atoms, 1)
        h2 = x_norm[:, None, :, :].repeat(1, n_atoms, 1, 1)
        if self.input_edges:
            (
                edge_shift_msa,
                edge_scale_msa,
                edge_gate_msa,
                edge_shift_mlp,
                edge_scale_mlp,
                edge_gate_mlp,
            ) = self.adaLN_edge_modulation(t_emb).chunk(6, dim=1)
            edge_attr_norm = modulate(
                self.input_adaln_norm_edge(edge_attr * edge_mask.unsqueeze(-1)),
                edge_shift_msa.unsqueeze(1),
                edge_scale_msa.unsqueeze(1),
            )
            input_features = torch.cat([h1, h2, edge_attr_norm, dist], dim=-1)
        else:
            input_features = torch.cat([h1, h2, dist], dim=-1)

        messages = self.feature_embedder(input_features * edge_mask.unsqueeze(-1))
        x_norm = torch.sum(messages, dim=-2)  # * mask.unsqueeze(-1)
        # import ipdb; ipdb.set_trace()
        # QKV projection
        qkv = self.qkv_proj(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q, K = self.norm_q(Q * mask.unsqueeze(-1)), self.norm_k(K * mask.unsqueeze(-1))
        # Q, K = self.norm_q(Q), self.norm_k(K)

        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=self.num_heads)
        Q, K, V = map(reshaper, (Q, K, V))

        # Assume n_atoms is the sequence length
        # Create a causal mask (upper triangular matrix with -inf above the diagonal)
        # import ipdb; ipdb.set_trace()
        # attn_mask = edge_mask.bool().unsqueeze(1) #! having rows as all false causes issues for now will be upated in pytroch 2.5? https://github.com/pytorch/pytorch/pull/133882
        attn_mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1).bool().unsqueeze(1)
        # float_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float32).to(attn_mask.device)
        # float_attn_mask.masked_fill_(~attn_mask, torch.finfo(torch.float32).min)
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        attn_output = einops.rearrange(attn_output, "b h s d -> b s (h d)")
        y = self.out_projection(attn_output)
        # import ipdb; ipdb.set_trace()
        # Gated Residual
        x = x + gate_msa.unsqueeze(1) * y
        # Feed Forward
        x = (
            x
            + gate_mlp.unsqueeze(1)
            * self.ffn(modulate(self.ff_norm_inner(x * mask.unsqueeze(-1)), shift_mlp, scale_mlp))
        ) * mask.unsqueeze(-1)
        if self.output_edges:
            (
                edge_shift_msa,
                edge_scale_msa,
                edge_gate_msa,
                edge_shift_mlp,
                edge_scale_mlp,
                edge_gate_mlp,
            ) = self.adaLN_edge_modulation(t_emb).chunk(6, dim=1)
            y1 = y[:, :, None, :].repeat(1, 1, n_atoms, 1)
            y2 = y[:, None, :, :].repeat(1, n_atoms, 1, 1)
            edge_input = y1 + y2
            # edge = edge_attr + edge_gate_msa.unsqueeze(1).unsqueeze(2) * self.lin_edge0(edge_input) #! edge_attr is None here
            edge = edge_gate_msa.unsqueeze(1).unsqueeze(2) * self.lin_edge0(edge_input)
            e_in = self.lin_edge1(torch.cat([edge, dist], dim=-1))
            edge_attr = (
                edge
                + edge_gate_mlp.unsqueeze(1).unsqueeze(2)
                * self.ffn_edge(
                    modulate(
                        self.ff_norm_inner_edge(e_in * edge_mask.unsqueeze(-1)),
                        edge_shift_mlp.unsqueeze(1),
                        edge_scale_mlp.unsqueeze(1),
                    )
                )
            ) * edge_mask.unsqueeze(-1)
            return x, edge_attr
        else:
            return x, None


def coord2distfn(x, scale_dist_features=1, mask=None):
    # import ipdb; ipdb.set_trace()
    x = x * mask.unsqueeze(-1).unsqueeze(-1)
    rel_vec = x.unsqueeze(-4) - x.unsqueeze(-3)  # x.unsqueeze(-2) - x.unsqueeze(-3)  # [b, a, a, 3, K]
    radial = (rel_vec**2).sum(dim=3, keepdim=False)  # [b, a, a, K]
    if scale_dist_features >= 2:
        dotproduct = (x.unsqueeze(-3) * x.unsqueeze(-4)).sum(dim=-2, keepdim=False)  # shape [num_edges, 1]
        radial = torch.cat([radial, dotproduct], dim=-1)  # shape [num_edges, 2]
    if scale_dist_features == 4:
        #! TODO: this would be B x N x N x d_i repeated of dim=1 and sj for dim = 2 of just hte sum. skip for now
        p_i, p_j = x.unsqueeze(-4), x.unsqueeze(-3)
        d_i, d_j = (
            torch.pow(p_i, 2).sum(-2, keepdim=False).clamp(min=1e-6).sqrt(),
            torch.pow(p_j, 2).sum(-2, keepdim=False).clamp(min=1e-6).sqrt(),
        )
        radial = torch.cat([radial, d_i, d_j], dim=-1)

    return radial


class MDNoPyg(nn.Module):
    def __init__(
        self,
        num_layers=10,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=256,
        invariant_edge_feat_dim=256,
        atom_classes=16,
        edge_classes=5,
        num_heads=4,
        n_vector_features=128,
    ):
        super(MDNoPyg, self).__init__()
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.n_vector_features = n_vector_features
        self.atom_embedder = nn.Sequential(
            *[
                Linear(self.num_atom_classes, invariant_node_feat_dim),
                nn.SiLU(),
                Linear(invariant_node_feat_dim, invariant_node_feat_dim),
            ]
        )
        self.edge_embedder = nn.Sequential(
            *[
                Linear(self.num_edge_classes, invariant_edge_feat_dim),
                nn.SiLU(),
                Linear(invariant_edge_feat_dim, invariant_edge_feat_dim),
            ]
        )

        self.coord_emb = Linear(1, n_vector_features, bias=False)
        self.coord_pred = Linear(n_vector_features, 1, bias=False)
        self.scale_dist_features = 2
        #! TODO do we need coord prediction head which is mlp then 0 CoM?
        self.atom_type_head = PredictionHead(atom_classes, invariant_node_feat_dim)
        self.edge_type_head = PredictionHead(edge_classes, invariant_edge_feat_dim, edge_prediction=True)
        self.time_embedding = TimestepEmbedder(invariant_node_feat_dim)
        self.bond_refine = BondRefine(invariant_node_feat_dim, invariant_edge_feat_dim)
        self.dit_layers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.attention_mix_layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.dit_layers.append(
                DiTeBlock(
                    invariant_node_feat_dim,
                    invariant_edge_feat_dim,
                    num_heads=num_heads,
                    input_edges=i == 0,
                    output_edges=i == num_layers - 1,
                    scale_dist_features=self.scale_dist_features,
                )
            )
            self.egnn_layers.append(
                EGNNBlock(
                    invariant_node_feat_dim=invariant_node_feat_dim,
                    invariant_edge_feat_dim=invariant_edge_feat_dim,
                    n_vector_features=n_vector_features,
                    time_embedding_dim=invariant_node_feat_dim,
                    use_cross_product=True,
                    input_edges=i == num_layers - 1,
                )
            )

        self.node_blocks = nn.ModuleList(
            [Linear(invariant_node_feat_dim, invariant_node_feat_dim) for i in range(num_layers)]
        )
        # self.edge_blocks = nn.ModuleList(
        #     [Linear(invariant_edge_feat_dim, invariant_edge_feat_dim) for i in range(num_layers)]
        # )

    def forward(self, mask, edge_mask, X, H, E, t):
        pos = self.coord_emb(X.unsqueeze(-1))  # N x 3 x K
        H = self.atom_embedder(H)
        E = self.edge_embedder(E)
        te = self.time_embedding(t)
        edge_attr = E
        atom_hids = H
        # edge_hids = edge_attr
        for layer_index in range(len(self.dit_layers)):
            # import ipdb; ipdb.set_trace()
            distances = coord2distfn(pos, self.scale_dist_features, mask)  # E x K
            H, edge_attr = self.dit_layers[layer_index](mask, H, te, edge_attr, edge_mask, distances)
            pos, _ = self.egnn_layers[layer_index](mask, pos, H, edge_mask, edge_attr, te)  #! TODO at time here
            atom_hids = atom_hids + self.node_blocks[layer_index](H)
            # edge_hids = edge_hids + self.edge_blocks[layer_index](edge_attr)

        X = self.coord_pred(pos).squeeze(-1) * mask.unsqueeze(-1)
        x = zero_com(X, mask) * mask.unsqueeze(-1)
        H = atom_hids
        edge_attr = self.bond_refine(mask, x, H, edge_attr, edge_mask)
        h_logits, _ = self.atom_type_head(mask, H)
        e_logits, _ = self.edge_type_head.predict_edges(edge_mask, edge_attr)
        out = {
            "x_hat": x,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
            # "Z_hat": z_logits,
        }
        return out
