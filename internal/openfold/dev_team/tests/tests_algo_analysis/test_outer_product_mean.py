# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from internal.openfold.dev_team.algo_analysis.outer_product_mean_metrics import outer_product_mean_metrics


# EvoformerStack parameter set
N_sequences = 124
N_residues = 256
c_m = 256
c_hidden_opm = 32
c_z = 128


def test_outer_product_mean_metrics():
    m_shape = (N_sequences, N_residues, c_m)
    z_shape, outer_product_mean_metrics_out = outer_product_mean_metrics(m_shape, c_hidden_opm, c_z)
    mults_fwd_in_gsheet = 17_523_142_656
    adds_fwd_in_gsheet = 17_396_924_416
    memory_footprint_in_bytes_in_gsheet = 341_573_632
    params_in_gsheet = 148_160
    assert z_shape == (256, 256, 128)
    assert outer_product_mean_metrics_out.number_of_mults_fwd == mults_fwd_in_gsheet
    assert outer_product_mean_metrics_out.number_of_adds_fwd == adds_fwd_in_gsheet
    assert outer_product_mean_metrics_out.memory_footprint_in_bytes == memory_footprint_in_bytes_in_gsheet
    assert outer_product_mean_metrics_out.number_of_params == params_in_gsheet


test_outer_product_mean_metrics()
