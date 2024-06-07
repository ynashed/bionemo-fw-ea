import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean, scatter_softmax

from bionemo.model.molecule.moco.models.mpnn import MLP


class PredictionHead(nn.Module):
    def __init__(self, num_classes, feat_dim, discrete=True):
        super().__init__()
        self.num_classes = num_classes
        self.discrete = discrete
        self.projection = MLP(feat_dim, feat_dim, num_classes)

    def forward(self, H, batch):
        logits = self.projection(H)
        if self.discrete:
            probs = scatter_softmax(logits, index=batch, dim=0, dim_size=H.size(0))
        else:
            probs = H - scatter_mean(H, index=batch, dim=0)[batch]
        return logits, probs


class LossFunction(nn.Module):
    def __init__(self, discrete_class_weight=None):
        super().__init__()
        self.f_continuous = nn.MSELoss(reduction='none')  # can also use HuberLoss
        if discrete_class_weight is None:
            self.f_discrete = nn.CrossEntropyLoss(reduction='none')
        else:
            self.f_discrete = nn.CrossEntropyLoss(weight=discrete_class_weight, reduction='none')
            #! We can up weight certain bonds to make sure this is correct

    def forward(self, batch, logits, data, weight=None, continuous=True):
        batch_size = len(batch.unique())
        if continuous:
            loss = self.f_continuous(logits, data)
            output = logits
        else:
            loss = self.f_discrete(logits, data)
            output = torch.argmax(logits, dim=-1)
        loss = scatter_mean(loss, index=batch, dim=0, dim_size=batch_size)
        if weight:
            loss = loss * weight
        loss = loss.mean()
        return loss, output

    def distance_loss(self, batch, X_true, X_pred, Z_pred):
        true_distance = torch.tensor([], device=X_true.device)
        x_pred_distance = torch.tensor([], device=X_true.device)
        z_pred_distance = torch.tensor([], device=X_true.device)
        batch_size = len(batch.unique())
        c_batch = []
        for element in range(batch_size):
            x_true = X_true[batch == element]
            x_pred = X_pred[batch == element]
            c_batch.extend([element] * x_true.size(0) * x_true.size(0))
            dist = torch.cdist(x_true, x_true).flatten()
            dist_pred = torch.cdist(x_pred, x_pred).flatten()
            dist_z = Z_pred[batch == element][:, batch == element].flatten()
            true_distance = torch.cat([true_distance, dist], dim=-1)
            x_pred_distance = torch.cat([x_pred_distance, dist_pred], dim=-1)
            z_pred_distance = torch.cat([z_pred_distance, dist_z], dim=-1)
        c_batch = torch.Tensor(c_batch).to(torch.int64).to(X_true.device)
        A = self.f_continuous(true_distance, x_pred_distance)
        B = self.f_continuous(true_distance, z_pred_distance)
        C = self.f_continuous(x_pred_distance, z_pred_distance)
        A = scatter_mean(A, c_batch, dim=0, dim_size=batch_size).mean()
        B = scatter_mean(B, c_batch, dim=0, dim_size=batch_size).mean()
        C = scatter_mean(C, c_batch, dim=0, dim_size=batch_size).mean()
        return A, B, C


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
                # import ipdb; ipdb.set_trace()
                adj_matrix[idx][jdx] = torch.nn.functional.one_hot(torch.randint(0, 5, (1,)), 5).squeeze(0)
    # print(adj_matrix)

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

    source, target = E_idx
    E = A[source, target]  # E x 5
    # E_all = torch.cat((d.unsqueeze(1), a.unsqueeze(1), r_norm, E), dim=-1)  # E x 10
    edge_embedder = nn.Linear(5, 32)
    E = edge_embedder(E.float())

    loss_function = LossFunction()
    # import ipdb; ipdb.set_trace()
    out = loss_function.distance_loss(batch_ligand, X, X, Z.sum(-1))
