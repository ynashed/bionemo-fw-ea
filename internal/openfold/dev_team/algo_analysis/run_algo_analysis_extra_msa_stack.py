# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#
#   title: run_algo_analysis_extra_msa_stack.py
#   description:
#
#   notes:


# %%


import seaborn as sns

from internal.openfold.dev_team.algo_analysis.extra_msa_stack_metrics import (
    extra_msa_stack_metrics,
)


sns.set()
# %% Set parameters
#   -ultimate these parameters can be ready directly from conf yaml file
#

# ExtraMsaStack parameter set
num_residues = 256  # model.train_sequence_crop_size
num_sequences = 1024  # model.max_extra_msa
c_e = 64  # model.extra_msa_stack_config.c_e
c_z = 128  # model.extra_msa_stack_config.c_z
c_hidden_msa_att = 8
c_hidden_tri_att = 32
c_hidden_tri_mul = 128
c_hidden_opm = 32
num_heads_msa = 8
num_heads_tri = 4
transition_n = 4
num_extra_msa_blocks = 4


e_shape = (num_sequences, num_residues, c_e)
z_shape = (num_residues, num_residues, c_z)

# %%
#

extra_msa_stack_shape, extra_msa_stack_metrics = extra_msa_stack_metrics(
    e_shape,
    z_shape,
    c_hidden_msa_att=c_hidden_msa_att,
    c_hidden_tri_att=c_hidden_tri_att,
    c_hidden_tri_mul=c_hidden_tri_mul,
    c_hidden_opm=c_hidden_opm,
    num_heads_msa=num_heads_msa,
    num_extra_msa_blocks=num_extra_msa_blocks,
    num_heads_tri=num_heads_tri,
    transition_n=transition_n,
)


# %%
print(extra_msa_stack_metrics)

print("run_algo_analysis_extra_msa_stack.py, end")
