import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_softmax

from bionemo.model.molecule.moco.models.mpnn import MLP


class AttentionLayer(nn.Module):
    def __init__(
        self,
        equivariant_node_feature_dim=3,
        invariant_node_feat_dim=64,
        invariant_edge_feat_dim=32,
        num_heads=4,
    ):
        super().__init__()
        self.equivariant_node_feature_dim = equivariant_node_feature_dim
        self.invariant_node_feat_dim = invariant_node_feat_dim
        self.invariant_edge_feat_dim = invariant_edge_feat_dim
        self.num_heads = num_heads
        self.dist_proj = MLP(equivariant_node_feature_dim, invariant_node_feat_dim, invariant_node_feat_dim)
        self.KV = MLP(
            invariant_node_feat_dim + 1, 2 * invariant_node_feat_dim, 2 * invariant_node_feat_dim * self.num_heads
        )
        self.Q = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_node_feat_dim * self.num_heads)
        self.pair_bias = MLP(invariant_node_feat_dim, invariant_node_feat_dim, 1 * self.num_heads)
        self.gate = MLP(invariant_edge_feat_dim, invariant_edge_feat_dim, 1 * self.num_heads, last_act='sigmoid')
        self.phi_h = MLP(invariant_node_feat_dim * self.num_heads, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_x = MLP(invariant_node_feat_dim * self.num_heads, invariant_node_feat_dim, 1)
        self.coor_update_clamp_value = 10.0
        self.left_z = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_node_feat_dim)
        self.right_z = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_node_feat_dim)
        self.joint_z = MLP(invariant_node_feat_dim, invariant_node_feat_dim, invariant_node_feat_dim)
        self.phi_e = MLP(
            2 * invariant_node_feat_dim + equivariant_node_feature_dim + self.num_heads,
            invariant_edge_feat_dim,
            invariant_edge_feat_dim,
        )
        # self.reset_parameters()

    def forward(self, batch, X, H, E, E_idx, Z):
        # Compute Q, K, V
        # import ipdb; ipdb.set_trace()
        src, dst = E_idx
        Q = self.Q(H[src])
        rel_dist = ((X[src] - X[dst]) ** 2).sum(dim=-1, keepdim=True)
        kv_input = torch.cat([rel_dist, H[dst]], dim=-1)
        K, V = self.KV(kv_input).split(self.num_heads * self.invariant_node_feat_dim, dim=-1)

        num_edges = E.size(0)
        num_nodes = H.size(0)
        head_dim = self.invariant_node_feat_dim
        Q = Q.view(num_edges, self.num_heads, head_dim)  # Shape: [num_edges, num_heads, head_dim]
        K = K.view(num_edges, self.num_heads, head_dim)  # Shape: [num_edges, num_heads, head_dim]
        V = V.view(num_edges, self.num_heads, head_dim)  # Shape: [num_edges, num_heads, head_dim]
        B = self.pair_bias(Z).view(num_nodes, num_nodes, self.num_heads, 1)

        # Attention scores
        attention_scores = (Q * K).sum(dim=-1) / (head_dim**0.5)  # Shape: [num_edges, num_heads]
        attention_scores = attention_scores + (B[src, dst].squeeze(2))  # Adding the bias term
        attention_scores = scatter_softmax(attention_scores, src, dim=0)  # Shape: [num_edges, num_heads]
        alpha_ij = self.gate(E) * attention_scores
        #! Verify direction of scatter and source and dist

        # Apply attention to values
        attention_output = alpha_ij.unsqueeze(-1) * V  # [src]  # Shape: [num_edges, num_heads, head_dim]
        attention_output = scatter(
            attention_output, dst, dim=0, dim_size=num_nodes, reduce='sum'
        )  # Shape: [num_nodes, num_heads, head_dim]
        # H update
        H_out = H + self.phi_h(attention_output.view(num_nodes, -1))  # Average over the attention heads
        # X update
        coor_wij = self.phi_x(V.view(num_edges, -1))  # E x 3
        if self.coor_update_clamp_value:
            coor_wij.clamp_(min=-self.coor_update_clamp_value, max=self.coor_update_clamp_value)
        rel_coors = X[src] - X[dst]
        X_rel_norm = rel_coors / (1 + torch.sqrt(rel_dist + 1e-8))
        x_update = scatter(
            alpha_ij.mean(1).unsqueeze(1) * X_rel_norm * coor_wij, index=dst, dim=0, reduce='sum', dim_size=X.shape[0]
        )
        X_out = X + x_update
        # Z update
        alpha = scatter(alpha_ij.mean(1), index=dst, dim=0, reduce='sum', dim_size=X.shape[0]).unsqueeze(1)
        z_update = self.joint_z(alpha * torch.einsum("...ik,...jk->...ijk", self.left_z(H), self.right_z(H)))
        Z_out = Z + z_update
        # E update
        # import ipdb; ipdb.set_trace()
        e_inputs = torch.cat([alpha_ij, X_rel_norm, H[src], H[dst]], dim=-1)
        e_update = self.phi_e(e_inputs)
        E_out = E + e_update

        return H_out, X_out, Z_out, E_out


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
    # r = X[target] - X[source]  # E x 3
    # a = X[target] * X[source]
    # a = a.sum(-1)  # E
    # d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
    # d = d.sqrt()  # E
    # r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))  # E x 3
    E = A[source, target]  # E x 5
    # E_all = torch.cat((d.unsqueeze(1), a.unsqueeze(1), r_norm, E), dim=-1)  # E x 10
    edge_embedder = nn.Linear(5, 32)
    E = edge_embedder(E.float())
    import ipdb

    ipdb.set_trace()
    # model = AttentionLayer()
    # print(X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())
    # for i in range(25):
    #     H, X, Z, E = model(batch_ligand, X, H, E, E_idx, Z)
    #     print(X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())

    from bionemo.model.molecule.moco.models.mpnn import EquivariantMessagePassingLayer

    mpnn = EquivariantMessagePassingLayer()  #! Layer norm forces stable.
    model = AttentionLayer(num_heads=10)
    print(X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())
    for i in range(25):
        X, H, E, m_ij = mpnn(batch_ligand, X, H, E_idx, E)
        print("MPNN", X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())
        H, X, Z, E = model(batch_ligand, X, H, E, E_idx, Z)
        print("ATTN", X.sum().item(), H.sum().item(), E.sum().item(), Z.sum().item())
    print("MPNN Params", sum(p.numel() for p in mpnn.parameters() if p.requires_grad))
    print("ATTN Params", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Success")
