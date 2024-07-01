# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from internal.openfold.dev_team.algo_analysis.triangle_metrics import triangle_multplication


# EvoformerStack parameter set
N_sequences = 124
N_residues = 256
c = 128
c_z = 128


def test_triangle_metrics():
    z_shape = (N_residues, N_residues, c_z)
    z_shape, triangle_multplication_metrics_out = triangle_multplication(z_shape, c)
    mults_fwd_in_gsheet = 8_674_344_960
    adds_fwd_in_gsheet = 8_648_654_848
    memory_footprint_in_bytes_in_gsheet = 402_653_184
    params_in_gsheet = 99_328
    assert z_shape == (256, 256, 128)
    assert abs(triangle_multplication_metrics_out.number_of_params - params_in_gsheet) / params_in_gsheet < 0.05
    assert (
        abs(triangle_multplication_metrics_out.number_of_mults_fwd - mults_fwd_in_gsheet) / mults_fwd_in_gsheet < 0.05
    )
    assert abs(triangle_multplication_metrics_out.number_of_adds_fwd - adds_fwd_in_gsheet) / adds_fwd_in_gsheet < 0.05
    assert (
        abs(triangle_multplication_metrics_out.memory_footprint_in_bytes - memory_footprint_in_bytes_in_gsheet)
        / memory_footprint_in_bytes_in_gsheet
        < 0.05
    )
