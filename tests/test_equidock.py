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
import pytest
import torch

from bionemo.model.protein.equidock.loss_metrics.eval import rigid_transform_Kabsch_3D, rigid_transform_Kabsch_3D_torch
from bionemo.model.protein.equidock.loss_metrics.intersection_loss import compute_body_intersection_loss
from bionemo.model.protein.equidock.model.utils import compute_cross_attention


torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.mark.needs_gpu
@pytest.mark.parametrize("data_type", [torch.float32, torch.float64])
def test_Kabsch_algorithm(data_type):
    for _ in range(10):
        # Random points
        A = np.random.randn(3, 10)
        B = np.random.randn(3, 10)

        # Get transformations from both functions
        R_numpy, t_numpy = rigid_transform_Kabsch_3D(A, B)
        R_torch, t_torch = rigid_transform_Kabsch_3D_torch(
            torch.from_numpy(A).to('cuda').to(data_type), torch.from_numpy(B).to('cuda').to(data_type)
        )

        # Convert the torch tensors to numpy arrays for easy comparison
        R_torch = R_torch.detach().cpu().numpy()
        t_torch = t_torch.detach().cpu().numpy()

        # Assert if matrices and vectors are close
        assert np.allclose(
            R_numpy, R_torch, atol=1e-4 if data_type == torch.float32 else 1e-6
        ), f"Rotation matrices differ: \n{R_numpy}\n{R_torch}"
        assert np.allclose(
            t_numpy, t_torch, atol=1e-4 if data_type == torch.float32 else 1e-6
        ), f"Translation vectors differ: \n{t_numpy}\n{t_torch}"


def test_body_intersection_loss():
    model_ligand_coors_deform = torch.zeros(1, 3)
    bound_receptor_repres_nodes_loc_array = 5.0 * torch.ones(1, 3)
    sigmas = torch.linspace(1.0, 10.0, 10)
    old_res = compute_body_intersection_loss(
        model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigmas[0], 1.0
    )

    for sigma in sigmas[1:]:
        new_res = compute_body_intersection_loss(
            model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, 1.0
        )
        if new_res < old_res:
            assert False, "Body intersection loss should be increasing with sigma increase!"
        old_res = new_res


@pytest.mark.parametrize("cross_msgs", [True, False])
def test_cross_attention(cross_msgs: bool):
    N = 21
    M = 11
    D = 16
    d = 4

    q = np.random.randn(N, D)
    k = np.random.randn(M, D)
    v = np.random.randn(M, d)
    mask = np.random.randint(0, 2, size=(N, M))

    a = mask * (q @ k.transpose([1, 0])) - 1000.0 * (1.0 - mask)

    a_x = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
    if cross_msgs:
        expected = a_x @ v
    else:
        expected = (
            q * 0
        )  # TODO: should it be v * 0? The paper has q*0, but maybe that's just because q.shape[1:] == v.shape[1:]?

    q = torch.from_numpy(q)
    k = torch.from_numpy(k)
    v = torch.from_numpy(v)
    mask = torch.from_numpy(mask).bool()

    result = compute_cross_attention(q, k, v, mask, cross_msgs).detach().cpu().numpy()

    assert np.allclose(expected, result, atol=1e-4), f"Cross message attention fails: \n{expected}\n{result}"
