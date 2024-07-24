# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from model.molecule.moco.models.utils import PredictionHead
from torch import nn
from torch_scatter import scatter_mean

from bionemo.model.molecule.moco.arch.scratch.attention import AttentionLayer
from bionemo.model.molecule.moco.arch.scratch.mpnn import AdaLN, EquivariantMessagePassingLayer, TimestepEmbedder


class EquivariantTransformerBlock(nn.Module):
    def __init__(
        self, equivariant_node_feature_dim=3, invariant_node_feat_dim=64, invariant_edge_feat_dim=32, num_heads=4
    ):
        super().__init__()
        self.mpnn = EquivariantMessagePassingLayer(
            equivariant_node_feature_dim, invariant_node_feat_dim, invariant_edge_feat_dim
        )
        self.mha = AttentionLayer(
            equivariant_node_feature_dim, invariant_node_feat_dim, invariant_edge_feat_dim, num_heads
        )

    def forward(self, batch, X, H, E_idx, E, Z):
        #  print(X.shape, H.shape, E.shape, E_idx.shape, Z.shape)
        X, H, E, m_ij = self.mpnn(batch, X, H, E_idx, E)
        #  print(X.shape, H.shape, E.shape, E_idx.shape, Z.shape)
        X, H, E, Z = self.mha(batch, X, H, E_idx, E, Z)
        X = X - scatter_mean(X, batch, dim=0)[batch]
        return X, H, E, Z


class MoCo(nn.Module):
    def __init__(
        self,
        atom_classes=16,
        atom_features=64,
        edge_classes=5,
        edge_features=32,
        extra_discrete_classes=None,
        num_layers=5,
        num_heads=10,
    ):
        super(MoCo, self).__init__()
        if extra_discrete_classes:
            num_extra_discrete = sum(extra_discrete_classes.values())
        else:
            num_extra_discrete = 0
        self.atom_embedder = nn.Linear(atom_classes + num_extra_discrete, atom_features)
        self.edge_embedder = nn.Linear(edge_classes, edge_features)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(EquivariantTransformerBlock(3, atom_features, edge_features, num_heads))
        self.layers = nn.ModuleList(self.layers)
        self.time_embedding = TimestepEmbedder(atom_features)
        self.ada_ln = AdaLN(atom_features, atom_features)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        #! TODO do we need coord prediction head which is mlp then 0 CoM?
        self.atom_type_head = PredictionHead(atom_classes + num_extra_discrete, atom_features)
        self.edge_type_head = PredictionHead(edge_classes, edge_features, edge_prediction=True)
        self.distance_head = PredictionHead(1, atom_features, distance_prediction=True)
        # if num_extra_discrete > 0:
        #     self.discrete_pred_heads = {}
        #     for key, dim in extra_discrete_classes.items():
        #         self.discrete_pred_heads[key] = PredictionHead(dim, atom_features)
        self.num_extra_discrete = num_extra_discrete
        # self.confidence_model = nn.Sequential([MLP(atom_features, 4*atom_features, 4*atom_features), nn.LayerNorm(4*atom_features), MLP(4*atom_features, 4*atom_features, 1, last_act='sigmoid')])

    def forward(self, batch, X, H, E_idx, E, t):
        # import ipdb; ipdb.set_trace()
        #! Now we assume the data is always 1 hot inputs
        H = self.atom_embedder(H)
        E = self.edge_embedder(E)
        te = self.time_embedding(t, batch)  # B x D
        H = self.ada_ln(H, te, batch)  # N x D
        Z = H.unsqueeze(1) * H.unsqueeze(0)
        # print(X.shape, H.shape, E.shape, E_idx.shape, Z.shape)
        # print(X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())
        for layer in self.layers:
            X, H, E, Z = layer(batch, X, H, E_idx, E, Z)
            # print(X.shape, H.shape, E.shape, E_idx.shape, Z.shape)
            # print(X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())
        h_logits, h_prob = self.atom_type_head(batch, H)  #! These values look weird H values blown up
        e_logits, e_prob = self.edge_type_head.predict_edges(batch, E, E_idx)
        z_logits = self.distance_head.predict_distances(batch, Z)
        # confidence = self.confidence_model(H)
        out = {
            "x_hat": X,
            "h_logits": h_logits,
            "edge_attr_logits": e_logits,
            "Z_hat": z_logits,
            # "confidence": confidence
        }
        # if self.num_extra_discrete > 0:
        #     for key, predictor in self.discrete_pred_heads.items():
        #         out

        return out
