import torch.nn.functional as F
from torch import nn

from bionemo.model.molecule.moco.models.mpnn import AdaLN, EquivariantMessagePassingLayer, TimestepEmbedder
from bionemo.model.molecule.moco.models.utils import PredictionHead


class MoCo(nn.Module):
    def __init__(self, atom_classes=16, atom_features=64, edge_classes=5, edge_features=32, num_layers=2):
        super(MoCo, self).__init__()
        self.atom_embedder = nn.Linear(atom_classes, atom_features)
        self.edge_embedder = nn.Linear(edge_classes, atom_features)
        self.layers = nn.ModuleList(
            [
                EquivariantMessagePassingLayer(3, atom_features, edge_features),
                EquivariantMessagePassingLayer(3, atom_features, edge_features),
            ]
        )
        self.time_embedding = TimestepEmbedder(atom_features)
        self.ada_ln = AdaLN(atom_features, atom_features)
        self.num_atom_classes = atom_classes
        self.num_edge_classes = edge_classes
        self.atom_type_head = PredictionHead(atom_classes, atom_features)
        self.edge_type_head = PredictionHead(edge_classes, edge_features)

    def forward(self, batch, X, H, E_idx, E, t):
        H = self.atom_embedder(F.one_hot(H, self.num_atom_classes).float())
        E = self.edge_embedder(F.one_hot(E, self.num_edge_classes).float())
        te = self.time_embedding(t, batch)  # B x D
        H = self.ada_ln(H, te, batch)  # error
        for layer in self.layers:
            X, H, E, m_ij = layer(batch, X, H, E_idx, E)

        h_logits, h_pred = self.atom_type_head(batch, H)
        e_logits, e_pred = self.atom_type_head(batch, H)
        return {
            "x_hat": X,
            "h_hat": h_logits,
            "h_pred": h_pred,
            "edge_attr_hat": e_logits,
            "edge_attr_pred": e_pred,
        }
