import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean

from bionemo.model.molecule.moco.models.attention import AttentionLayer
from bionemo.model.molecule.moco.models.mpnn import AdaLN, EquivariantMessagePassingLayer, TimestepEmbedder
from bionemo.model.molecule.moco.models.utils import PredictionHead


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
    def __init__(self, atom_classes=16, atom_features=64, edge_classes=5, edge_features=32, num_layers=10):
        super(MoCo, self).__init__()
        self.atom_embedder = nn.Linear(atom_classes, atom_features)
        self.edge_embedder = nn.Linear(edge_classes, edge_features)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(EquivariantTransformerBlock(3, atom_features, edge_features, 10))
        self.layers = nn.ModuleList(self.layers)
        self.time_embedding = TimestepEmbedder(atom_features)
        self.ada_ln = AdaLN(atom_features, atom_features)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.atom_type_head = PredictionHead(atom_classes, atom_features)
        self.edge_type_head = PredictionHead(edge_classes, edge_features, edge_prediction=True)
        # self.distance_head = PredictionHead(1, atom_features, distance_prediction=True)

    def forward(self, batch, X, H, E_idx, E, t):
        H = self.atom_embedder(F.one_hot(H, self.num_atom_classes).float())
        E = self.edge_embedder(F.one_hot(E, self.num_edge_classes).float())
        te = self.time_embedding(t, batch)  # B x D
        H = self.ada_ln(H, te, batch)  # N x D
        Z = H.unsqueeze(1) * H.unsqueeze(0)
        # print(X.shape, H.shape, E.shape, E_idx.shape, Z.shape)
        # print(X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())
        for layer in self.layers:
            X, H, E, Z = layer(batch, X, H, E_idx, E, Z)
            # print(X.shape, H.shape, E.shape, E_idx.shape, Z.shape)
            # print(X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())
        h_logits, h_pred = self.atom_type_head(batch, H)  #! These values look weird H values blown up
        e_logits, e_pred = self.edge_type_head.predict_edges(batch, E, E_idx)
        z_logits = None  # self.distance_head.predict_distances(batch, Z)
        return {
            "x_hat": X,
            "h_hat": h_logits,
            "h_pred": h_pred,
            "edge_attr_hat": e_logits,
            "edge_attr_pred": e_pred,
            "Z_hat": z_logits,
        }
