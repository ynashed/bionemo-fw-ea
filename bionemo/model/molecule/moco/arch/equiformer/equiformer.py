# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Literal

import torch
import torch_geometric
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_scatter import scatter

from bionemo.model.molecule.moco.arch.equiformer.drop import EquivariantDropout, GraphDropPath
from bionemo.model.molecule.moco.arch.equiformer.fast_activation import Activation, Gate
from bionemo.model.molecule.moco.arch.equiformer.fast_layer_norm import EquivariantLayerNormFast
from bionemo.model.molecule.moco.arch.equiformer.gaussian_rbf import GaussianRadialBasisLayer
from bionemo.model.molecule.moco.arch.equiformer.graph_norm import EquivariantGraphNorm
from bionemo.model.molecule.moco.arch.equiformer.instance_norm import EquivariantInstanceNorm
from bionemo.model.molecule.moco.arch.equiformer.layer_norm import EquivariantLayerNormV2
from bionemo.model.molecule.moco.arch.equiformer.radial_func import RadialProfile
from bionemo.model.molecule.moco.arch.equiformer.tensor_product_rescale import (
    FullyConnectedTensorProductRescale,
    LinearRS,
    TensorProductRescale,
    irreps2gate,
    sort_irreps_even_first,
)


_RESCALE = True
_USE_BIAS = True

# QM9
_MAX_ATOM_TYPE = 5
# Statistics of QM9 with cutoff radius = 5
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666


def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)


def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        norm_layer='graph',
    ):
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.norm = get_norm_layer(norm_layer)(self.irreps_out)

    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        return out


class FullyConnectedTensorProductRescaleNormSwishGate(FullyConnectedTensorProductRescaleNorm):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        norm_layer='graph',
    ):
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
        super().__init__(
            irreps_in1,
            irreps_in2,
            gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
            norm_layer=norm_layer,
        )
        self.gate = gate

    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        out = self.gate(out)
        return out


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
        super().__init__(
            irreps_in1,
            irreps_in2,
            gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.gate = gate

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out


def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, internal_weights=False, bias=True):
    '''
    The irreps of output is pre-determined.
    `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(
        irreps_node_input,
        irreps_edge_attr,
        irreps_output,
        instructions,
        internal_weights=internal_weights,
        shared_weights=internal_weights,
        bias=bias,
        rescale=_RESCALE,
    )
    return tp


class SeparableFCTP(torch.nn.Module):
    '''
    Use separable FCTP for spatial convolution.
    '''

    def __init__(
        self,
        irreps_node_input,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        use_activation=False,
        norm_layer='graph',
        internal_weights=False,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)

        self.dtp = DepthwiseTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_output,
            bias=False,
            internal_weights=internal_weights,
        )

        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k

        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)

        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars,
                    [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates,
                    # gates (scalars)
                    [torch.sigmoid for _, ir in irreps_gates],
                    irreps_gated,  # gated tensors
                )
            self.gate = gate

    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        '''
        Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by
        self.dtp_rad(`edge_scalars`).
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out


@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
    Reshape vectors of shape [N, irreps_mid] to vectors of shape
    [N, num_heads, irreps_head].
    '''

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(self.__class__.__name__, self.irreps_head, self.num_heads)


@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
    Convert vectors of shape [N, num_heads, irreps_head] into
    vectors of shape [N, irreps_head * num_heads].
    '''

    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


class ConcatIrrepsTensor(torch.nn.Module):
    def __init__(self, irreps_1, irreps_2):
        super().__init__()
        assert irreps_1 == irreps_1.simplify()
        self.check_sorted(irreps_1)
        assert irreps_2 == irreps_2.simplify()
        self.check_sorted(irreps_2)

        self.irreps_1 = irreps_1
        self.irreps_2 = irreps_2
        self.irreps_out = irreps_1 + irreps_2
        self.irreps_out, _, _ = sort_irreps_even_first(self.irreps_out)  # self.irreps_out.sort()
        self.irreps_out = self.irreps_out.simplify()

        self.ir_mul_list = []
        lmax = max(irreps_1.lmax, irreps_2.lmax)
        irreps_max = []
        for i in range(lmax + 1):
            irreps_max.append((1, (i, -1)))
            irreps_max.append((1, (i, 1)))
        irreps_max = o3.Irreps(irreps_max)

        start_idx_1, start_idx_2 = 0, 0
        dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(irreps_2)
        for _, ir in irreps_max:
            dim_1, dim_2 = None, None
            index_1 = self.get_ir_index(ir, irreps_1)
            index_2 = self.get_ir_index(ir, irreps_2)
            if index_1 != -1:
                dim_1 = dim_1_list[index_1]
            if index_2 != -1:
                dim_2 = dim_2_list[index_2]
            self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
            start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
            start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2

    def get_irreps_dim(self, irreps):
        muls = []
        for mul, ir in irreps:
            muls.append(mul * ir.dim)
        return muls

    def check_sorted(self, irreps):
        lmax = None
        p = None
        for _, ir in irreps:
            if p is None and lmax is None:
                p = ir.p
                lmax = ir.l
                continue
            if ir.l == lmax:
                assert p < ir.p, 'Parity order error: {}'.format(irreps)
            assert lmax <= ir.l

    def get_ir_index(self, ir, irreps):
        for index, (_, irrep) in enumerate(irreps):
            if irrep == ir:
                return index
        return -1

    def forward(self, feature_1, feature_2):
        output = []
        for i in range(len(self.ir_mul_list)):
            start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
            if mul_1 is not None:
                output.append(feature_1.narrow(-1, start_idx_1, mul_1))
            if mul_2 is not None:
                output.append(feature_2.narrow(-1, start_idx_2, mul_2))
        output = torch.cat(output, dim=-1)
        return output

    def __repr__(self):
        return '{}(irreps_1={}, irreps_2={})'.format(self.__class__.__name__, self.irreps_1, self.irreps_2)


@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)
    '''

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)

        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads)  # irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify()
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha))  # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()

        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                self.irreps_pre_attn,
                fc_neurons,
                use_activation=True,
                norm_layer=None,
                internal_weights=False,
            )
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_heads,
                fc_neurons=None,
                use_activation=False,
                norm_layer=None,
                internal_weights=True,
            )
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)), num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_all,
                fc_neurons,
                use_activation=False,
                norm_layer=None,
            )
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), num_heads
            )

        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)

        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot)  # Following GATv2

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input, drop_prob=proj_drop)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, **kwargs):
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]

        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)

            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))

        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        attention_scores = alpha
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)

        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree

        node_output = self.proj(attn)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output, attention_scores

    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str


@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
    Use two (FCTP + Gate)
    '''

    def __init__(self, irreps_node_input, irreps_node_attr, irreps_node_output, irreps_mlp_mid=None, proj_drop=0.0):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, bias=True, rescale=_RESCALE
        )
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, bias=True, rescale=_RESCALE
        )

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, drop_prob=proj_drop)

    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


@compile_mode('script')
class TransBlock(torch.nn.Module):
    '''
    1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
    2. Use pre-norm architecture
    '''

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        irreps_mlp_mid=None,
        norm_layer='layer',
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None else self.irreps_node_input

        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ga = GraphAttention(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=self.rescale_degree,
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)

        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop,
        )

        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output, bias=True, rescale=_RESCALE
            )

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, batch=None, **kwargs):
        node_output = node_input
        node_features = node_input
        node_features = self.norm_1(node_features, batch=batch)
        # norm_1_output = node_features
        node_features, attention_scores = self.ga(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
        )

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch=None)
        node_output = node_output + node_features

        node_features = node_output
        node_features = self.norm_2(node_features, batch=None)
        # node_features = self.concat_norm_output(norm_1_output, node_features)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch=None)
        node_output = node_output + node_features

        return node_output, attention_scores


class ResidueIrrepsEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_residue_embedding, irreps_node_input, bias=True):
        super().__init__()
        self.irreps_residue_embedding = o3.Irreps(irreps_residue_embedding)
        self.residue_irreps = LinearRS(irreps_node_input, self.irreps_residue_embedding, bias=bias)

    def forward(self, irreps_features):
        irreps_features = self.residue_irreps(irreps_features)

        return irreps_features


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0

    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num**0.5)
        return out

    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)


class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(
            irreps_node_embedding, irreps_edge_attr, irreps_node_embedding, internal_weights=False, bias=False
        )
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for slice, slice_sqrt_k in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0, dim_size=node_features.shape[0])
        return node_features


class EquivariantTransformer(torch.nn.Module):
    def __init__(
        self,
        hidden_size=32,
        irreps_node_embedding='128x0e+64x1e+32x2e',
        num_layers=2,
        irreps_node_attr='1x0e',
        irreps_sh='1x0e+1x1e+1x2e',
        max_radius=20.0,
        number_of_basis=128,
        # choose between rbf or combining rbf
        basis_type: Literal['gaussian', 'edge_attr'] = 'gaussian',
        fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e',
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer

        # '{}x0e+{}x1e'.format(hidden_size, hidden_size)
        self.irreps_in = '{}x0e'.format(hidden_size)  # x1e
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(self.irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax

        self.irreps_feature = self.irreps_node_embedding
        self.num_layers = num_layers
        self.irreps_edge_attr = (
            o3.Irreps(irreps_sh) if irreps_sh is not None else o3.Irreps.spherical_harmonics(self.lmax)
        )

        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.residue_embed = ResidueIrrepsEmbeddingNetwork(self.irreps_node_embedding, self.irreps_node_input)

        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
            self.fc_neurons = [self.number_of_basis] + fc_neurons

        elif self.basis_type == 'edge_attr':
            self.rbf = lambda x: torch.cat(x, dim=-1)
            self.fc_neurons = [hidden_size * 2 + 1] + fc_neurons

        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding, self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE
        )

        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer,
            )
            self.blocks.append(blk)

        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)

        self.head = LinearRS(self.irreps_feature, o3.Irreps('{}x0e+1x1e'.format(hidden_size)), rescale=_RESCALE)

        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    def forward(self, pos, hidden_states, edge_index, edge_attr=None, te=None):
        # convert residue embedding into irreps representation
        B, H = hidden_states.size()
        residue_irreps_embedding = self.residue_embed(hidden_states)

        # pos = pos.reshape(B, 3)

        edge_dst, edge_src = edge_index

        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr, x=edge_vec, normalize=True, normalization='component'
        )

        edge_length = edge_vec.norm(dim=1)
        if self.basis_type == 'gaussian':
            edge_length_embedding = self.rbf(edge_length)  # edge_vec, edge_attr, t_e
        else:
            assert edge_attr is not None
            assert te is not None
            edge_length_embedding = self.rbf([edge_attr, te, edge_length.unsqueeze(-1)])

        edge_degree_embedding = self.edge_deg_embed(
            residue_irreps_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst
        )
        node_features = residue_irreps_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        for blk in self.blocks:
            node_features, _ = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=None,
            )

        # TODO: do we need attention scores?!
        node_features = self.norm(node_features, batch=None)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        outputs = self.head(node_features)
        hidden_states, dT = outputs[:, :H], outputs[:, H:]

        # TODO: Update pos with dT and remove mean!
        return pos + dT, hidden_states


def create_batch_pairwise_matrix(
    num_batches, num_nodes_per_batch, source_indices, target_indices, edge_features, pair_batch_indices, device
):
    pairwise_matrix = torch.zeros(
        (num_batches, num_nodes_per_batch, num_nodes_per_batch), dtype=edge_features.dtype
    ).to(device)

    batch_offsets = pair_batch_indices * num_nodes_per_batch
    source_indices = source_indices - batch_offsets
    target_indices = target_indices - batch_offsets

    pairwise_matrix.index_put_((pair_batch_indices, source_indices, target_indices), edge_features, accumulate=True)
    pairwise_matrix.index_put_((pair_batch_indices, target_indices, source_indices), edge_features, accumulate=True)

    return pairwise_matrix


def equivariant_transformer_l2(
    hidden_size: int,
    radius: float = 16,
    basis_type: Literal['gaussian', 'edge_attr'] = 'edge_attr',
    num_basis: int = 128,
    num_layers: int = 2,
    **kwargs,
):
    model = EquivariantTransformer(
        hidden_size=hidden_size,
        irreps_node_embedding='128x0e+64x1e+32x2e',
        num_layers=num_layers,
        irreps_node_attr='1x0e',
        irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis,
        basis_type=basis_type,
        fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e',
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
    )

    return model


if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as R

    from bionemo.utils.tests import Deterministic

    with Deterministic():
        model = equivariant_transformer_l2(32)
        # model = model.eval()
        # mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]).bool()

        hidden_state = torch.randn(7, 32)  # * mask.unsqueeze(-1)
        transitions = torch.randn(7, 3)  # * mask.unsqueeze(-1)
        edge_index = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6],
                [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 6, 4, 6, 4, 5],
            ]
        )
        # transitions = transitions * mask.unsqueeze(-1)
        # transitions = transitions - torch.sum(transitions, dim=-2, keepdim=True) / torch.sum(
        #     mask.unsqueeze(-1), dim=-2, keepdim=True
        # )
        # transitions = transitions * mask.unsqueeze(-1)

        print(transitions)
        rotation = R.random().as_matrix()
        rotation_tensor = torch.tensor(rotation, dtype=torch.float32)

        x = model(hidden_state, transitions, edge_index)
        x_rotated = model(hidden_state, torch.matmul(transitions, rotation_tensor.T), edge_index)
        torch.testing.assert_close(x_rotated[-1], torch.matmul(x[-1], rotation_tensor.T), atol=1e-5, rtol=1e-5)

        torch.testing.assert_close(x_rotated[0], x[0], atol=1e-5, rtol=1e-5)

        for y in x:
            print(y.shape)