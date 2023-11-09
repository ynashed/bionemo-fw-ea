# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal

import torch
from e3nn import o3
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch_scatter import scatter


if "USE_FAST_TP" in os.environ:
    from bionemo.model.molecule.diffdock.utils.fast_tensor_product import FullyConnectedTP as FastTP

from nemo.core import NeuralModule

from bionemo.model.molecule.diffdock.utils.batchnorm import BatchNorm
from bionemo.model.molecule.diffdock.utils.tensor_product import (
    TPFirst,
    TPForth,
    TPSecond,
    TPThird,
)


CUSTOM_TP_MAPPING = dict(enumerate([TPFirst, TPSecond, TPThird, TPForth]))


class AtomEncoder(NeuralModule):
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type=None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == "esm":
                self.lm_embedding_dim = 1280
            else:
                raise ValueError(
                    "LM Embedding type was not correctly determined. LM embedding type: ",
                    self.lm_embedding_type,
                )
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(
                x[
                    :,
                    self.num_categorical_features : self.num_categorical_features + self.num_scalar_features,
                ]
            )
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim :]], axis=1))
        return x_embedding


class TensorProductAPI(nn.Module):
    def __init__(
        self,
        in_irreps,
        sh_irreps,
        out_irreps,
        tp_type: Literal['e3nn', 'marta'] = "e3nn",
        tp_param: int = -1,
        dtype: torch.dtype = torch.float32,
    ):
        super(TensorProductAPI, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.tp_type = tp_type
        self.tp_param = tp_param
        self.dtype = dtype

        self.tp = self.setup(in_irreps, sh_irreps, out_irreps, tp_type, tp_param, dtype)
        self.weight_numel = self.tp.weight_numel

    @staticmethod
    def setup(in_irreps, sh_irreps, out_irreps, tp_type, tp_param, dtype):
        if "USE_FAST_TP" in os.environ:
            tp = FastTP(str(in_irreps), str(sh_irreps), str(out_irreps), dtype=dtype)
        elif tp_type == "marta":
            tp = torch.jit.script(CUSTOM_TP_MAPPING[tp_param]())
        elif tp_type == "e3nn":
            tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        return tp

    def forward(self, *args, **kwargs):
        return self.tp(*args, **kwargs)

    def __getstate__(self):
        return {
            "in_irreps": self.in_irreps,
            "out_irreps": self.out_irreps,
            "sh_irreps": self.sh_irreps,
            "tp_type": self.tp_type,
            "tp_param": self.tp_param,
            "dtype": self.dtype,
        }

    def __setstate__(self, d):
        super(TensorProductAPI, self).__init__()
        self.in_irreps = d["in_irreps"]
        self.out_irreps = d["out_irreps"]
        self.sh_irreps = d["sh_irreps"]
        self.tp_type = d["tp_type"]
        self.tp_param = d["tp_param"]
        self.dtype = d["dtype"]
        self.tp = self.setup(**d)
        self.weight_numel = self.tp.weight_numel


class TensorProductConvLayer(NeuralModule):
    def __init__(
        self,
        in_irreps,
        sh_irreps,
        out_irreps,
        n_edge_features,
        residual=True,
        batch_norm=True,
        batch_norm_with_shift=True,
        dropout=0.0,
        hidden_features=None,
        tp_type: str = "e3nn",
        tp_param: int = -1,
        dtype: torch.dtype = torch.float32,
    ):
        """Performs graph convolution with tensor product.

        For calculating tensor product we use FullyConnectedTensorProduct
        from in_irreps and sh_irreps to out_irreps, using learnable weights.

        For a set of configurations of inputs and outputs used in the
        TensorProductScoreModel, a custom implementation can be used.
        In order to use this option, provide tp_type="marta" with tp_param=i where i is the index of
        the irreps used in TensorProductScoreModel. If this option is
        selected, in_irreps, sh_irreps, and out_irreps are ignored.
        """
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.batch_norm_with_shift = batch_norm_with_shift
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = TensorProductAPI(in_irreps, sh_irreps, out_irreps, tp_type, tp_param, dtype=dtype)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, self.tp.weight_numel),
        )
        self.batch_norm = BatchNorm(out_irreps, with_shift=batch_norm_with_shift) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce="mean"):
        edge_src, edge_dst = edge_index
        edge_attr = self.fc(edge_attr)

        if "USE_FAST_TP" in os.environ:
            tp = self.tp(node_attr[edge_dst], edge_sh, edge_attr)
        else:
            with autocast(enabled=False):
                inputs = (inp.float() for inp in (node_attr[edge_dst], edge_sh, edge_attr))
                tp = self.tp(*inputs)

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class GaussianSmearing(NeuralModule):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
