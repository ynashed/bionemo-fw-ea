# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


# This test suit checks if we can load original, public weights into BioNeMo.
# Particularly, if we turn on initial-training optimizations like weights fusing,
# is it still possible to load it. We also include sanity-checking for
# scenarios that should fail.
#
# OpenFold in BioNeMo has some layers renamed so we always have to map them when
# we load .pt checkpoints. This is reflected in
# bionemo.model.protein.openfold.checkpoint_utils.remap_layers_names

import os

import omegaconf
import pytest
import torch

from bionemo.model.protein.openfold.checkpoint_utils import load_pt_checkpoint
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.optim_hub import OptimHub


BIONEMO_HOME = os.getenv("BIONEMO_HOME")
CHECKPOINT_PATH = os.path.join(BIONEMO_HOME, "models/protein/openfold/initial_training.pt")


@pytest.fixture(scope="module")
def base_config():
    # base config is enough. It contains architectural details for initial-training without
    # training hyperparameters
    return omegaconf.OmegaConf.load(os.path.join(BIONEMO_HOME, "examples/protein/openfold/conf/base_config.yaml"))


@pytest.fixture(scope="function")
def alphafold_model(base_config, request):
    if hasattr(request, 'param'):
        OptimHub.enable_multiple(request.param)
    return AlphaFold(base_config.model, None)


@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_load_openfold_fail_direct(alphafold_model):
    """
    We should be NOT able to load pt checkpoint directly as some layers have different names.
    """
    with pytest.raises(RuntimeError):
        alphafold_model.load_state_dict(torch.load(CHECKPOINT_PATH))


@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
@pytest.mark.parametrize(
    'alphafold_model',
    [[], ['mha_fused_gemm'], ['mha_fused_gemm', 'mha_triton'], ['mha_fused_gemm', 'mha_triton', 'layernorm_inductor']],
    indirect=['alphafold_model'],
)
def test_load_openfold_mapping(alphafold_model):
    """
    We should be able to load pt checkpoint if GEMM fusing from MLPerf is turned on.
    """
    load_pt_checkpoint(alphafold_model, CHECKPOINT_PATH)
