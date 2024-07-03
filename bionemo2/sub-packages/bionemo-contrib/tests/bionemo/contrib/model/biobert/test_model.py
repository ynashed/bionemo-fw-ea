# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import tarfile

import torch
from torch.nn import functional as F

from bionemo.contrib.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.contrib.model.biobert.model import BioBertConfig, BiobertSpecOption
from bionemo.contrib.testing import megatron_parallel_state_utils
from bionemo.contrib.utils.dtypes import get_autocast_dtype
from bionemo.contrib.utils.weight_utils import nemo1_to_nemo2_biobert_key_mapping


# TODO(@jstjohn) use fixtures for pulling down data and checkpoints
# python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss
test_script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
bionemo2_root = test_script_dir.parent.parent.parent.parent.parent.parent.parent
nemo1_checkpoint_path = bionemo2_root / "models/singlecell/geneformer/geneformer-10M-240530.nemo"
data_path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/processed_data"


def test_bionemo2_rootdir():
    assert (bionemo2_root / "sub-packages").exists(), "Could not find bionemo2 root directory."
    assert (bionemo2_root / "sub-packages").is_dir(), "sub-packages is supposed to be a directory."


def test_weight_shapes_match(seed: int = 42):
    autocast_dtype = get_autocast_dtype("bf16-mixed")
    geneformer_config = BioBertConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=2048,
        fp32_residual_connection=False,  # TODO(@jstjohn) check this
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=True,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,  # TODO(@jstjohn) check this
        fp16_lm_cross_entropy=False,
        params_dtype=autocast_dtype,
        pipeline_dtype=autocast_dtype,
        autocast_dtype=autocast_dtype,  # setting this speeds things up a lot
        gradient_accumulation_fusion=False,  # THIS BREAKS STUFF, leave False
        layernorm_zero_centered_gamma=False,  # TODO(@jstjohn) check this
        layernorm_epsilon=1.0e-12,
        activation_func=F.relu,  # TODO(@jstjohn) check this
        qk_layernorm=True,  # TODO(@jstjohn) check this
        apply_residual_connection_post_layernorm=False,  # False is new default, True was BERT pub.
        bias_activation_fusion=False,  # TODO(@jstjohn) check this
        bias_dropout_fusion=True,  # TODO(@jstjohn) check this
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=True,  # This has to be set to True if we use the mixed precision plugin
        biobert_spec_option=BiobertSpecOption.bert_layer_local_spec,
    )
    data_error_str = "Please download test data with:\n`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    data_dir = pathlib.Path(data_path)
    train_data_path = data_dir / "train"
    if not nemo1_checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {nemo1_checkpoint_path}. {data_error_str}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_error_str}")

    with tarfile.open(
        nemo1_checkpoint_path, "r"
    ) as old_ckpt, torch.no_grad(), megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        ckpt_file = old_ckpt.extractfile("./model_weights.ckpt")
        old_weights = torch.load(ckpt_file)
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {'tokenizer': tokenizer, 'median_dict': _}:
                pass
            case _:
                assert False
        new_model = geneformer_config.configure_model(tokenizer)
        new_state_dict = new_model.state_dict_for_save_checkpoint()
        # Set the new_model_prefix to "" since we are looking at the base megatron model and not the lightning module which stores a copy of
        #  this model into self.module
        old_keys = {nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="") for k in old_weights.keys()}
        assert len(old_keys) == len(old_weights.keys()), "Mapping unexpectedly discarded some keys."
        new_keys = set(new_state_dict.keys())
        for k, v in old_weights.items():
            # Make sure the shapes of the weights match.
            assert new_state_dict[nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="")].shape == v.shape
        extra_keys = new_keys - old_keys
        extra_non_null_keys = {k for k in extra_keys if new_state_dict[k] is not None}
        assert not extra_non_null_keys, "There are new keys that have state that is missing from the old checkpoint."
        missing_old_keys = old_keys - new_keys
        assert not missing_old_keys, "There are keys in the old checkpoint that are missing from the new model."
