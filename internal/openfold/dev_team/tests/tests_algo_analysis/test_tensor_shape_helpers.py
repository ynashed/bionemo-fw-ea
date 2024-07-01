# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from internal.openfold.dev_team.algo_analysis.tensor_shape_helpers import num_elements_excluding


def test_000():
    # EvoformerStack parameter set
    num_residues = 256
    c_z = 128  # residude-residue pair_embedding_dim

    z_shape = (num_residues, num_residues, c_z)
    out = num_elements_excluding(z_shape, excluded_axes=[-1])

    # assert on results
    expected_num = num_residues * num_residues
    assert expected_num == out
