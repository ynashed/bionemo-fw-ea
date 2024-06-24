# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import List

import pytest
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.identity_op import IdentityOp

from bionemo.contrib.model.biobert import transformer_specs
from bionemo.contrib.model.layers import TELayerNorm


def test_enum_str_choices():
    options: List[str] = [o.value for o in transformer_specs.BiobertSpecOption]
    for o_str in options:
        # Make sure argparse will be happy with the string equality
        assert o_str == transformer_specs.BiobertSpecOption(o_str)
        if o_str != "random_string":
            # Show that some random string doesn't match
            assert "random_string" != transformer_specs.BiobertSpecOption(o_str)


def test_enum_equality():
    assert (
        transformer_specs.BiobertSpecOption('bert_layer_local_spec')
        == transformer_specs.BiobertSpecOption.bert_layer_local_spec
    )


def test_local_spec_sets_qk_ln():
    spec_with_qk = transformer_specs.get_biobert_spec(
        transformer_specs.BiobertSpecOption.bert_layer_local_spec_with_qk_ln, qk_layernorm=True
    )
    spec_no_qk = transformer_specs.get_biobert_spec(
        transformer_specs.BiobertSpecOption.bert_layer_local_spec_with_qk_ln, qk_layernorm=False
    )
    assert spec_with_qk.submodules.self_attention.submodules.q_layernorm == FusedLayerNorm
    assert (
        spec_with_qk.submodules.self_attention.submodules.q_layernorm
        == spec_with_qk.submodules.self_attention.submodules.k_layernorm
    )
    assert spec_no_qk.submodules.self_attention.submodules.q_layernorm == IdentityOp
    assert (
        spec_no_qk.submodules.self_attention.submodules.q_layernorm
        == spec_no_qk.submodules.self_attention.submodules.k_layernorm
    )


def test_te_spec_sets_qk_ln():
    spec_with_qk = transformer_specs.get_biobert_spec(
        transformer_specs.BiobertSpecOption.bert_layer_with_transformer_engine_and_qk_ln_spec, qk_layernorm=True
    )
    spec_no_qk = transformer_specs.get_biobert_spec(
        transformer_specs.BiobertSpecOption.bert_layer_with_transformer_engine_and_qk_ln_spec, qk_layernorm=False
    )
    assert spec_with_qk.submodules.self_attention.submodules.q_layernorm == TELayerNorm
    assert (
        spec_with_qk.submodules.self_attention.submodules.q_layernorm
        == spec_with_qk.submodules.self_attention.submodules.k_layernorm
    )
    assert spec_no_qk.submodules.self_attention.submodules.q_layernorm == IdentityOp
    assert (
        spec_no_qk.submodules.self_attention.submodules.q_layernorm
        == spec_no_qk.submodules.self_attention.submodules.k_layernorm
    )


def test_get_spec_bad_input():
    with pytest.raises(NotImplementedError):
        transformer_specs.get_biobert_spec("bad_input")
