# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch_scatter import scatter, scatter_add


def scale_prior(prior, batch, num_atoms, c=0.2):
    return c * prior * np.log(num_atoms + 1)[batch]


def equivariant_ot_prior(self, data_chunk, batch):
    """Permute the from_mols batch so that it forms an approximate mini-batch OT map with to_mols"""
    #! prior has to be as big as the largets input and then we throw stuff away
    #! noise_batch is a list of beath elements of max atom num x 3
    batch_size = int(max(batch) + 1)
    data_batch = [data_chunk[batch == idx] for idx in range(batch_size)]
    max_num_atoms = max([x.shape[0] for x in data_batch])
    noise_batch = [self.prior(max_num_atoms) for _ in range(batch_size)]
    mol_matrix = []
    cost_matrix = []

    # Create matrix with data on outer axis and noise on inner axis
    for data in data_batch:
        best_noise = [permute_and_slice(noise, data) for noise in noise_batch]
        sub_batch = torch.arange(len(noise_batch)).repeat_interleave(data.shape[0])
        best_noise = align_structures(torch.cat(best_noise, dim=0), sub_batch, data, broadcast_reference=True)
        best_noise = best_noise.reshape((len(noise_batch), data.shape[0], 3))
        best_costs = pairwise_distances(
            best_noise, data.repeat(len(noise_batch), 1).reshape(len(noise_batch), data.shape[0], 3)
        )[
            :, 0
        ]  # B x 1
        mol_matrix.append(best_noise)  # B x N x 3
        cost_matrix.append(best_costs.numpy())

    row_indices, col_indices = linear_sum_assignment(np.array(cost_matrix))
    optimal_noise = [mol_matrix[r][c] for r, c in zip(row_indices, col_indices)]
    return torch.cat(optimal_noise, dim=0)  #! returns N tot x 3 where this matches data_chunk


def pairwise_distances(tensor1, tensor2):
    # tensor1 and tensor2 are of shape (B, N, 3)
    B, N, _ = tensor1.shape

    # Expand tensors for broadcasting
    tensor1_exp = tensor1.unsqueeze(1)  # Shape: (B, 1, N, 3)
    tensor2_exp = tensor2.unsqueeze(0)  # Shape: (1, B, N, 3)

    # Calculate the squared differences and sum over the last dimension (3)
    distances_squared = torch.sum((tensor1_exp - tensor2_exp) ** 2, dim=-1)  # Shape: (B, B, N)

    # Sum the distances over the N dimension to get the final cost matrix
    cost_matrix = torch.sum(distances_squared, dim=-1)  # Shape: (B, B)

    return cost_matrix


def permute_and_slice(noise, data):
    N = noise.shape[0]
    M = data.shape[0]
    assert M <= N
    noise_indices = torch.arange(M)
    noise = noise[noise_indices, :]
    cost_matrix = torch.cdist(noise, data) ** 2
    _, noise_indices = linear_sum_assignment(cost_matrix.numpy())
    noise = noise[noise_indices, :]
    return noise


def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes]


@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0)

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices
