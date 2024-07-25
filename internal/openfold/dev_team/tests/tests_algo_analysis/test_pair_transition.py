# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from internal.openfold.dev_team.algo_analysis.transition import (
    pair_transition_metrics,
)


def test_pair_transition():
    c_z = 128
    N_residues = 256
    input_shape = (N_residues, N_residues, c_z)
    transition_n = 4
    pair_transition_shape, pair_transition_metrics_out = pair_transition_metrics(in_shape=input_shape, n=transition_n)

    gsheet_number_of_mults_fwd = 8_615_362_560
    gsheet_number_of_adds_fwd = 8_623_489_024
    gsheet_number_of_params = 131_968
    memory_footprint_in_bytes_in_gsheet = 201_326_592
    assert input_shape == pair_transition_shape
    assert gsheet_number_of_params == pair_transition_metrics_out.number_of_params
    assert gsheet_number_of_mults_fwd == pair_transition_metrics_out.number_of_mults_fwd
    assert gsheet_number_of_adds_fwd == pair_transition_metrics_out.number_of_adds_fwd
    assert memory_footprint_in_bytes_in_gsheet == pair_transition_metrics_out.memory_footprint_in_bytes
