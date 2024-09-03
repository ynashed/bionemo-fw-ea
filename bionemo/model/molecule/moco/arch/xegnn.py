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
from typing import List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm  # this function uses np.random internally
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from torch import Size, Tensor
from torch.nn import LayerNorm, Module


_shape_t = Union[int, List[int], Size]


class LayerNormNoBias(LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True):
        super(LayerNormNoBias, self).__init__(normalized_shape, eps, elementwise_affine, bias=False)

    def reset_parameters(self) -> None:
        self.weight.data.fill_(1.0)


# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0.0, scale=1.0)
TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978


class Linear(nn.Linear):
    """
    Linear transformation with extra non-standard initializations.

    Args:
        in_features: Last dimension of the input tensor.
        out_features: Last dimension of the output tensor.
        bias: Whether to learn an additive bias. Default: `True`.
        init: Parameter initialization method.
            One of:
            - "default": LeCun (fan-in) with a truncated normal distribution
            - "relu": He initialization with a truncated normal distribution
            - "glorot": fan-average Glorot uniform initialization
            - "gating": Weights=0, Bias=1
            - "normal": Normal initialization with std=1/sqrt(fan_in)
            - "final": Weights=0, Bias=0

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
        gain_scale: float = 0.001,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        # By default, the biases of the Linear layers are filled with zeros.
        if bias:
            self.bias.data.fill_(0.0)

        if init == "default":
            lecun_normal_init_(self.weight.data, gain_scale)
        elif init == "relu":
            he_normal_init_(self.weight.data, gain_scale)
        elif init == "glorot":
            glorot_uniform_init_(self.weight.data, gain_scale)
        elif init == "gating":
            gating_init_(self.weight.data)
            if bias:
                self.bias.data.fill_(1.0)
        elif init == "normal":
            normal_init_(self.weight.data)
        elif init == "final":
            final_init_(self.weight.data)
        else:
            raise ValueError(f"unknown init {repr(init)}")


def lecun_normal_init_(weight_data: torch.Tensor, scale: float) -> None:
    trunc_normal_init_(weight_data, scale=scale)


def he_normal_init_(weight_data: torch.Tensor, scale: float) -> None:
    trunc_normal_init_(weight_data, scale=scale)


def glorot_uniform_init_(weight_data: torch.Tensor, gain: float) -> None:
    nn.init.xavier_uniform_(weight_data, gain=gain)


def final_init_(weight_data: torch.Tensor) -> None:
    weight_data.fill_(0.0)


def gating_init_(weight_data: torch.Tensor) -> None:
    weight_data.fill_(0.0)


def normal_init_(weight_data: torch.Tensor) -> None:
    nn.init.kaiming_normal_(weight_data, nonlinearity="linear")


def trunc_normal_init_(
    weight_data: torch.Tensor,
    scale: float = 1.0,
    fan: str = "fan_in",
) -> None:
    assert isinstance(weight_data, torch.Tensor)
    assert not isinstance(weight_data, nn.Parameter)
    weight_shape = weight_data.shape
    weight_numel = weight_data.numel()
    fan_value = _calculate_fan(weight_shape, fan)
    scale = scale / max(1, fan_value)
    stddev = math.sqrt(scale) / TRUNCATED_NORMAL_STDDEV_FACTOR
    values = truncnorm.rvs(a=-2, b=2, loc=0, scale=stddev, size=weight_numel)
    values = np.reshape(values, newshape=weight_shape)
    weight_data.copy_(torch.tensor(values, device=weight_data.device))


def _calculate_fan(linear_weight_shape: torch.Size, fan: str = "fan_in") -> int:
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        fan_value = fan_in
    elif fan == "fan_out":
        fan_value = fan_out
    elif fan == "fan_avg":
        fan_value = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")
    return fan_value


def remove_mean_with_mask(x: Tensor, mask: Tensor):
    """
    Remove mean of x based on provided mask, and set masked elements to zero.

    Args:
        x (torch.Tensor): A tensor of shape [bs, n, d], where `bs` is the batch size,
                          `n` is the number of nodes, and `d` is the feature dimension.
        mask (torch.Tensor): A boolean tensor of shape [bs, n] indicating which
                                  elements to include in the mean computation.

    Returns:
        torch.Tensor: A tensor of the same shape as `x` with the mean removed for
                      the unmasked elements, and masked elements set to zero.
    """
    mask = mask.unsqueeze(-1)

    N = mask.sum(1, keepdims=True)

    x = x * mask

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * mask
    return x


class E3Norm(nn.Module):
    """
    E3Normalization based on mean of norm of positions

    Args:
        nn (_type_): _description_
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(2 * torch.ones((1, 1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    # [b, a, 3]  # [b, a]
    def forward(self, pos: torch.Tensor, mask: torch.Tensor = None):
        mask = mask.unsqueeze(-1)

        norm = torch.norm(pos * mask, dim=-1, keepdim=True)
        mean_norm = torch.sum(norm, dim=-2, keepdim=True) / torch.sum(mask, dim=-2, keepdim=True)

        new_pos = self.weight * pos / (mean_norm + self.eps)

        return new_pos * mask


class XEGNNBlock(Module):
    """
    X only EGNN
    """

    def __init__(
        self,
        invariant_node_feat_dim: int = 64,
        invariant_edge_feat_dim: int = 0,
        time_embedding_dim: int = 0,
        activation: Module = nn.SiLU(),
        use_clamp: bool = True,
        coords_range: float = 10.0,
        norm_constant: float = 1,
        aggregation_method: Literal['sum'] = 'sum',
        use_cross_product: bool = False,
    ):
        super(XEGNNBlock, self).__init__()

        self.use_clamp = use_clamp
        self.coors_range = coords_range

        self.message_in_dim = 2 * invariant_node_feat_dim + 1 + invariant_edge_feat_dim + time_embedding_dim

        self.h_norm = LayerNormNoBias(normalized_shape=(invariant_node_feat_dim), elementwise_affine=True)

        self.x_norm = E3Norm()

        self.phi_x = nn.Sequential(
            *[Linear(self.message_in_dim, self.message_in_dim), activation, Linear(self.message_in_dim, 1)]
        )

        self.use_cross_product = use_cross_product
        if self.use_cross_product:
            self.phi_x_cross = nn.Sequential(
                *[Linear(self.message_in_dim, self.message_in_dim), activation, Linear(self.message_in_dim, 1)]
            )

        self.norm_constant = norm_constant
        self.aggregation_method = aggregation_method

        if use_clamp:
            self.clamp = self._clamp
        else:
            self.clamp = nn.Identity()

    def _clamp(self, x):
        return torch.clamp(x, min=-self.coors_range, max=self.coors_range)

    def forward(self, x: Tensor, h: Tensor, mask: Tensor, t: Tensor, edge_attr: Tensor = None):
        """_summary_

        Args:
            x (Tensor): [b, a, 3]
            h (Tensor): [b, a, d_h]
            mask (Tensor): [b, a]
            t (Tensor): [b, d_t]
            edge_attr (Optiona[Tensor]): [b, a, a, d_e]

        Returns:
            Updated x position
        """
        bs, n_atoms, hdim = h.shape

        x = self.x_norm(x, mask)
        h = self.h_norm(h * mask.unsqueeze(-1))

        rel_vec = x.unsqueeze(-2) - x.unsqueeze(-3)  # [b, a, a, 3]
        rel_dist = (rel_vec**2).sum(dim=-1, keepdim=True)  # [b, a, a, 1]

        if edge_attr:
            edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)  # [b, a, a, 1 + d_e]
        else:
            edge_attr_feat = rel_dist

        h1 = h[:, :, None, :].repeat(1, 1, n_atoms, 1)
        h2 = h[:, None, :, :].repeat(1, n_atoms, 1, 1)

        h_message = torch.cat([h1, h2, edge_attr_feat], dim=-1)
        if t is not None:
            h_message = torch.cat([h_message, t.view(bs, 1, 1, -1).repeat(1, n_atoms, n_atoms, 1)], dim=-1)

        mask = mask.int()
        mask = (mask[:, :, None] + mask[:, None, :] > 1).float()

        mask = mask.unsqueeze(-1)

        x_updates_vals = self.clamp(self.phi_x(h_message) * mask)  # b, a, a, 1

        if self.use_cross_product:
            x_updates_cross = self.clamp(self.phi_x_cross(h_message) * mask)  # b, a, a, 1

        if self.aggregation_method == 'sum':
            x_update = torch.sum(
                x_updates_vals * rel_vec / (self.norm_constant + torch.sqrt(rel_dist + 1e-8)), dim=-2
            )  # b, a, 3
            if self.use_cross_product:
                rel_cross = torch.linalg.cross(x.unsqueeze(-2), x.unsqueeze(-3))  # b, a, a, 3
                x_update = x_update + (
                    (rel_cross) / (self.norm_constant + rel_cross.norm(dim=-1, keepdim=True)) * x_updates_cross
                ).sum(dim=-2)

        x = x + x_update

        return x, h


class EGNNBlock(Module):
    """
    X and H EGNN
    """

    def __init__(
        self,
        invariant_node_feat_dim: int = 64,
        invariant_edge_feat_dim: int = 0,
        time_embedding_dim: int = 0,
        activation: Module = nn.SiLU(),
        use_clamp: bool = True,
        coords_range: float = 10.0,
        norm_constant: float = 1,
        aggregation_method: Literal['sum'] = 'sum',
        use_cross_product: bool = False,
        attenion_type: Literal["None", "Taylor", "Simple"] = "Simple",
    ):
        super(EGNNBlock, self).__init__()

        self.use_clamp = use_clamp
        self.coors_range = coords_range

        self.invariant_node_feat_dim = invariant_node_feat_dim

        self.message_in_dim = 2 * invariant_node_feat_dim + 1 + invariant_edge_feat_dim + time_embedding_dim

        self.h_norm = LayerNormNoBias(normalized_shape=(invariant_node_feat_dim), elementwise_affine=True)

        self.x_norm = E3Norm()

        self.phi_x = nn.Sequential(
            *[Linear(self.message_in_dim, self.message_in_dim), activation, Linear(self.message_in_dim, 1)]
        )

        self.phi_e = nn.Sequential(
            *[
                Linear(self.message_in_dim, self.message_in_dim),
                activation,
                Linear(self.message_in_dim, invariant_node_feat_dim),
            ]
        )

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
        else:
            self.phi_att = TaylorSeriesLinearAttn(
                dim=self.invariant_node_feat_dim,
                dim_head=max(4, self.invariant_node_feat_dim // 2),
                heads=1,
                one_headed_kv=1,
                prenorm=True,
            )

        self.use_cross_product = use_cross_product
        if self.use_cross_product:
            self.phi_x_cross = nn.Sequential(
                *[Linear(self.message_in_dim, self.message_in_dim), activation, Linear(self.message_in_dim, 1)]
            )

        self.norm_constant = norm_constant
        self.aggregation_method = aggregation_method

        if use_clamp:
            self.clamp = self._clamp
        else:
            self.clamp = nn.Identity()

    def _clamp(self, x):
        return torch.clamp(x, min=-self.coors_range, max=self.coors_range)

    def forward(self, x: Tensor, h: Tensor, mask: Tensor, t: Tensor, edge_attr: Tensor = None):
        """_summary_

        Args:
            x (Tensor): [b, a, 3]
            h (Tensor): [b, a, d_h]
            mask (Tensor): [b, a]
            t (Tensor): [b, d_t]
            edge_attr (Optiona[Tensor]): [b, a, a, d_e]

        Returns:
            Updated x position
        """
        bs, n_atoms, hdim = h.shape

        x = self.x_norm(x, mask)
        h = self.h_norm(h * mask.unsqueeze(-1))

        rel_vec = x.unsqueeze(-2) - x.unsqueeze(-3)  # [b, a, a, 3]
        rel_dist = (rel_vec**2).sum(dim=-1, keepdim=True)  # [b, a, a, 1]

        if edge_attr:
            edge_attr_feat = torch.cat([edge_attr, rel_dist], dim=-1)  # [b, a, a, 1 + d_e]
        else:
            edge_attr_feat = rel_dist

        h1 = h[:, :, None, :].repeat(1, 1, n_atoms, 1)
        h2 = h[:, None, :, :].repeat(1, n_atoms, 1, 1)

        h_message = torch.cat([h1, h2, edge_attr_feat], dim=-1)  # b, a, a,  d_m
        if t is not None:
            h_message = torch.cat([h_message, t.view(bs, 1, 1, -1).repeat(1, n_atoms, n_atoms, 1)], dim=-1)

        mask = mask.int()
        mask = (mask[:, :, None] + mask[:, None, :] > 1).float()

        mask = mask.unsqueeze(-1)

        m_ij = self.phi_e(h_message * mask)  # b, a, a,  d_m

        if self.attenion_type == "Simple":
            att_val = self.phi_att(m_ij)  # b, a, a,  1
            m_ij = m_ij * att_val * mask  # b, a, a,  d_m
        elif self.attenion_type == "Taylor":
            m_ij = self.phi_att(m_ij.view(bs * n_atoms, n_atoms, -1), mask.view(bs * n_atoms, n_atoms).bool()).view(
                bs, n_atoms, n_atoms, -1
            )
        m_ij = torch.sum(m_ij * mask, dim=-2)  # b, a, d_h

        h_out = h + self.phi_h(torch.cat([h, m_ij], dim=-1))

        x_updates_vals = self.clamp(self.phi_x(h_message) * mask)  # b, a, a, 3

        if self.use_cross_product:
            x_updates_cross = self.clamp(self.phi_x_cross(h_message) * mask)

        if self.aggregation_method == 'sum':
            x_update = torch.sum(
                x_updates_vals * rel_vec / (self.norm_constant + torch.sqrt(rel_dist + 1e-8)), dim=-2
            )  # b, a, 3
            if self.use_cross_product:
                rel_cross = torch.linalg.cross(x.unsqueeze(-2), x.unsqueeze(-3))
                x_update = x_update + (
                    (rel_cross) / (self.norm_constant + rel_cross.norm(dim=-1, keepdim=True)) * x_updates_cross
                ).sum(dim=-2)

        x = x + x_update

        return x, h_out


class EGNN(Module):
    def __init__(
        self,
        num_layers: int = 4,
        invariant_node_feat_dim: int = 64,
        invariant_edge_feat_dim: int = 0,
        time_embedding_dim: int = 0,
        activation: Module = nn.SiLU(),
        use_clamp: bool = True,
        coords_range: float = 10.0,
        norm_constant=1,
        aggregation_method: Literal['sum'] = 'sum',
        use_cross_product=False,
        h_update: bool = True,
        attenion_type: Literal["None", "Taylor", "Simple"] = "Simple",
    ):
        super(EGNN, self).__init__()

        if not h_update:
            self.convs = nn.ModuleList(
                [
                    XEGNNBlock(
                        invariant_node_feat_dim,
                        invariant_edge_feat_dim,
                        time_embedding_dim,
                        activation,
                        use_clamp,
                        coords_range,
                        norm_constant,
                        aggregation_method,
                        use_cross_product,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    EGNNBlock(
                        invariant_node_feat_dim,
                        invariant_edge_feat_dim,
                        time_embedding_dim,
                        activation,
                        use_clamp,
                        coords_range,
                        norm_constant,
                        aggregation_method,
                        use_cross_product,
                        attenion_type,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(self, pos: Tensor, h: Tensor, mask: Tensor, t: Tensor, edge_attr: Tensor = None):
        """_summary_

        Args:
            pos (Tensor): [b, a, 3]
            h (Tensor): [b, a, d_h]
            mask (Tensor): [b, a]
            t (Tensor): [b, d_t]
            edge_attr (Optiona[Tensor]): [b, a, a, d_e]

        Returns:
            Updated position
        """
        pos = remove_mean_with_mask(pos, mask)

        for conv in self.convs:
            pos, h = conv(pos, h, mask, t, edge_attr)
            pos = remove_mean_with_mask(pos, mask)

        return pos, h


if __name__ == '__main__':
    torch.manual_seed(1382)
    bs = 256
    atom = 64
    h_dim = 16

    x = 20 * torch.randn(bs, atom, 3).cuda()
    h = 20 * torch.randn(bs, atom, h_dim).cuda()

    model = EGNN(4, h_dim, 0, h_update=True, attenion_type="Taylor").cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # x_hat = 20*torch.randn(bs, atom, 3).cuda()
    x_hat = x - x.mean(dim=-2, keepdim=True)
    mask = torch.ones(bs, atom).bool().cuda()

    h_hat = h

    for _ in range(100):
        x_out, h_out = model(x, h, mask, None)
        loss = ((x_hat - x_out) ** 2).mean() + ((h_hat - h_out) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    import time

    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(100):
        x_out, h_out = model(x, h, mask, None)
    torch.cuda.synchronize()
    e = time.perf_counter()
    print(e - s, loss.item())

    # print(h)
    x_out, h_out = model(x, h, mask, None)
    x_out, h_out = model(x_out, h_out, mask, None)

    print(h_out[0, 0])

    print(h[0, 0])

    print(x_out[0])
    print(x[0])

    exit()
