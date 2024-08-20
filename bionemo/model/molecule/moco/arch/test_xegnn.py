# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import torch
from scipy.spatial.transform import Rotation as R

from bionemo.bionemo.model.molecule.moco.arch.egnn import EGNN, E3Norm, remove_mean_with_mask
from bionemo.utils.tests import Deterministic


def test_e3norm():
    with Deterministic():
        rotation = R.random().as_matrix()
        rotation_tensor = torch.tensor(rotation, dtype=torch.float32)

        norm = E3Norm()
        x = torch.randn(5, 4, 3)
        mask = torch.randint(0, 2, (5, 4))
        mask = torch.sort(mask, descending=True)[0]
        mask[:, 0] = 1
        x_com = remove_mean_with_mask(x, mask)
        norm(x_com, mask)
        x_rotated = torch.matmul(x_com, rotation_tensor.T)

        torch.testing.assert_close(
            norm(x_rotated, mask), norm(torch.matmul(x_com, rotation_tensor.T), mask), atol=1e-5, rtol=1e-5
        )


def test_EGNN():
    with Deterministic():
        rotation = R.random().as_matrix()
        rotation_tensor = torch.tensor(rotation, dtype=torch.float32)

        x = torch.randn(5, 4, 3)
        h = torch.randn(5, 4, 10)
        mask = torch.randint(0, 2, (5, 4))
        mask = torch.sort(mask, descending=True)[0]
        mask[:, 0] = 1

        model = EGNN(12, 10, 0, use_cross_product=True)

        output, h_out = model(x, h, mask, None)

        x_rotated = torch.matmul(x, rotation_tensor.T)

        output_rotated, h_out_rotated = model(x_rotated, h, mask, None)

        torch.testing.assert_close(output_rotated, torch.matmul(output, rotation_tensor.T), atol=1e-5, rtol=1e-5)

        torch.testing.assert_close(h_out, h_out_rotated, atol=1e-5, rtol=1e-5)
