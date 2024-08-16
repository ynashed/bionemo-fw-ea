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
"""
How to adapt these tests:

1) Need to hook up our own pretraining workflow. In the code below, we do this via subproc and CLI. Is this still best practice?
    a) use the structure in sub-packages/bionemo-geneformer/tests/bionemo/geneformer/test_model.py:test_geneformer_nemo1_v_nemo2_inference_golden_values
    b) might need to look at utilities for setup/teardown to make sure the distributed stuff is handled correctly.
2) Need to inject the callbacks either via CLI or by manually inserting them here.
3) How do we want this to work for other modules? Lots of code could be duplicated here which makes it a challenge.
4) is this the right set of code to do this on?

"""

import math
import pathlib
from typing import Literal

import pytest
import torch
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import io, resume
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from torch.nn import functional as F

from bionemo import geneformer
from bionemo.core.utils.dtypes import get_autocast_dtype

# Do we want to re-export stuff in api?
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.testing.megatron_parallel_state_utils import clean_parallel_state_context
from bionemo.testing.testing_callbacks import (
    MetadataSaveCallback,
    RaiseAfterMetadataCallback,
    StopAndGoException,
    TestCheckpointIntegrityCallback,
)


bionemo2_root: pathlib.Path = (
    # geneformer module's path is the most dependable --> don't expect this to change!
    pathlib.Path(geneformer.__file__)
    # This gets us from 'sub-packages/bionemo-geneformer/src/bionemo/geneformer/__init__.py' to 'sub-packages/bionemo-geneformer'
    .parent.parent.parent.parent
    # From here, we want to get to the root of the repository: _before_ sub-packages/
    .parent.parent
).absolute()

assert bionemo2_root != pathlib.Path("/")
data_path: pathlib.Path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/processed_data"

MODEL_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"
USE_TE: bool = False  # TODO use this for high level decisions around whether we're ready to switch to TE


seq_length = 128  # NOTE(@skothenhill) decrease this if there are memory issues in CI


@pytest.fixture
def geneformer_config():
    # Facilitates running inside/outside pytest contexts.
    return _geneformer_config()


def _geneformer_config():
    """Setups the default geneformer config taken from pretrain.py. Update as needed."""
    autocast_dtype = get_autocast_dtype(MODEL_PRECISION)
    return GeneformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp16=autocast_dtype == torch.float16,
        bf16=autocast_dtype == torch.bfloat16,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=autocast_dtype,
        pipeline_dtype=autocast_dtype,
        autocast_dtype=autocast_dtype,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func=F.gelu,
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=BiobertSpecOption.bert_layer_local_spec.value,
        nemo1_ckpt_path=None,
        # Okay this is an inference only thing lol
        # return_only_hidden_states=True,  # This is what we did in nemo1 for inference
    )
    # BiobertSpecOption.bert_layer_with_transformer_engine_spec
    # if USE_TE
    # else BiobertSpecOption.bert_layer_local_spec.value,


def make_real_datamodule(tokenizer, seq_length, median_dict, devices, tensor_model_parallel_size, data_path):
    from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule

    num_dataset_workers = 0
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=data_path / "train",
        val_dataset_path=data_path / "val",
        test_dataset_path=data_path / "test",
        random_token_prob=0.1,  # this is the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=2,
        global_batch_size=2 * int(devices),  #  / tensor_model_parallel_size), TP is not involved here.
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )
    return data


# TODO (@skothenhill) How can we adapt this into a test harness?
def test_geneformer_stop_and_go(geneformer_config: GeneformerConfig, seed: int = 42):
    # Ensure the test data exists.
    data_error_str = "Please download test data with:\n`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    # Configuration stuff.
    data_dir = pathlib.Path(data_path)
    train_data_path = data_dir / "train"
    val_check_interval = 50
    lr = 1e-4
    num_steps = 200
    cosine_rampup_frac = 0.1
    cosine_hold_frac = 0.05
    root_dir = bionemo2_root / "results"
    devices = 1
    tp_size, pp_size = 1, 1

    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_error_str}")

    # Typical setup stuff. Do we need this with the test data? Find this out.
    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            pass
        case _:
            raise ValueError("Preprocessing must have failed.")

    # Taken from default argparse.
    optim = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=lr,
            optimizer="adam",
            use_distributed_optimizer=True,
        ),
        lr_scheduler=CosineAnnealingScheduler(
            max_steps=num_steps,
            min_lr=lr / 100,
            warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
            interval="step",
            monitor="reduced_train_loss",
            constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
        ),
    )

    module = BioBertLightningModule(
        config=geneformer_config,
        tokenizer=tokenizer,
    )
    data_module = make_real_datamodule(
        tokenizer, seq_length, median_dict, devices=devices, tensor_model_parallel_size=tp_size, data_path=data_path
    )

    # TODO (SKH) uh oh this isnt going to be liked by IOMixin
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_best_model=False,
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=val_check_interval,
        enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
        try_restore_best_ckpt=False,
    )

    # TODO: probably want to nuke this directory on startup everytime.
    exp_name = "geneformer_stopngo"
    # NOTE: PR 10090 makes it so you can set a version with resume_if_exists. In our case, we dont care about version, so we just ignore.
    nemo_logger: NeMoLogger = NeMoLogger(
        dir=str(root_dir),
        name=exp_name,
        use_datetime_version=False,
        version=None,
        tensorboard=None,
        wandb=None,
        ckpt=None,
    )
    metadata_dir = root_dir / exp_name

    # NOTE(SKH) Verified this is the same.
    with clean_parallel_state_context():
        tp_size = 1
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            ddp="megatron",
            find_unused_parameters=True,
            ckpt_include_optimizer=True,
            ckpt_parallel_save=False,
            ckpt_parallel_save_optim=False,
        )
        # NOTE(SKH) Verified this is the same.
        # NOTE(SKH) how do we consistently get the log directory? lots of magic happening.
        trainer = nl.Trainer(
            devices=devices,
            max_steps=num_steps,  # Hardcoded to debug
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=2,  # Hardcoded to coyp pretrain
            logger=False,
            val_check_interval=val_check_interval,
            num_nodes=1,
            callbacks=[
                io.track_io(MetadataSaveCallback)(
                    metadata_path=metadata_dir, metadata_keys=["learning_rate", "global_step"]
                ),
                io.track_io(RaiseAfterMetadataCallback)(metadata_path=metadata_dir),
                nl_callbacks.ModelCheckpoint(
                    save_best_model=False,
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=val_check_interval,
                    enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
                    try_restore_best_ckpt=False,
                ),
            ],  # is this what I need for stop and go?
            plugins=nl.MegatronMixedPrecision(precision=MODEL_PRECISION, amp_O2=False),
        )
        # io.track_io(LossLoggingCallback)()
        # Strategy has no attribute trainer.
        try:
            llm.train(
                model=module,
                data=data_module,
                trainer=trainer,
                log=nemo_logger,
                optim=optim,
                # NOTE (SKH) it seems like resume-if-exists isnt working how we expect.
                resume=resume.AutoResume(
                    path=None,  # Overrides the path found by resume_if_exists when set.
                    resume_if_exists=False,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                ),
            )
        except StopAndGoException:
            # Everything is as expected!
            ...

    # TODO: what actually needs to be torn down? strategy and trainer?
    print("Resetting.......")
    with clean_parallel_state_context():
        tp_size = 1
        pp_size = 1
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            ddp="megatron",
            find_unused_parameters=True,
            ckpt_include_optimizer=True,
        )
        # NOTE(SKH) Verified this is the same.
        trainer = nl.Trainer(
            devices=devices,
            max_steps=num_steps,  # Hardcoded to debug
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=2,  # Hardcoded to coyp pretrain
            val_check_interval=val_check_interval,
            num_nodes=1,
            callbacks=[
                TestCheckpointIntegrityCallback(
                    metadata_path=metadata_dir, metadata_keys=["global_step", "learning_rate"]
                ),
                checkpoint_callback,
            ],
            plugins=nl.MegatronMixedPrecision(precision=MODEL_PRECISION, amp_O2=False),
        )
        # LossLoggingCallback()
        llm.train(
            module,
            data_module,
            trainer,
            log=nemo_logger,
            optim=None,
            resume=resume.AutoResume(
                path=None,  # Overrides the path found by resume_if_exists when set.
                resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
            ),
        )


# Here is code that does the same thing for GPT.
def setup_gpt():
    import torch
    from megatron.core.optimizer import OptimizerConfig
    from nemo import lightning as nl
    from nemo.collections import llm

    seq_length = 128
    global_batch_size = 16

    ## setup the dummy dataset
    data = llm.MockDataModule(seq_length=seq_length, global_batch_size=global_batch_size, num_val_samples=32)

    ## initialize a small GPT model
    gpt_config = llm.GPTConfig(
        num_layers=6,
        hidden_size=384,
        ffn_hidden_size=1536,
        num_attention_heads=6,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
    )
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)

    ## initialize the strategy

    ## setup the optimizer

    # NOTE(SKH) Working arguments
    # optimizer="adam",
    # lr=6e-4,
    # bf16=True,
    lr = 1e-4
    opt_config = OptimizerConfig(
        lr = lr,
        optimizer="adam",
        use_distributed_optimizer=True,
    )
    num_steps = 500
    cosine_rampup_frac = 0.1
    cosine_hold_frac = 0.05
    opt = nl.MegatronOptimizerModule(config=opt_config,
        lr_scheduler=CosineAnnealingScheduler(
            max_steps=num_steps,
            min_lr=lr / 100,
            warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
            interval="step",
            monitor="reduced_train_loss",
            constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
        ),
    )
    # opt = nl.MegatronOptimizerModule(config=opt_config)
    exp_name = "gpt_stop_and_go"
    root_dir = bionemo2_root / "results"
    metadata_dir = root_dir / exp_name

    nemo_logger = nl.NeMoLogger(
        dir="test_logdir",  ## logs and checkpoints will be written here
    )
    nemo_logger: NeMoLogger = NeMoLogger(
        dir=str(root_dir),
        name=exp_name,
        use_datetime_version=False,
        version=None,
        tensorboard=None,
        wandb=None,
        ckpt=None,
    )
    with clean_parallel_state_context():
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_include_optimizer=True,
        )
        trainer = nl.Trainer(
            devices=1,  ## you can change the numebr of devices to suit your setup
            max_steps=500,
            accelerator="gpu",
            strategy=strategy,
            val_check_interval = 2,
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=False),
            callbacks=[
                io.track_io(MetadataSaveCallback)(
                    metadata_path=metadata_dir, metadata_keys=["learning_rate", "global_step"]
                ),
                io.track_io(RaiseAfterMetadataCallback)(metadata_path=metadata_dir),
                nl_callbacks.ModelCheckpoint(
                    save_best_model=False,
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=50,
                    enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
                    try_restore_best_ckpt=False,
                ),
            ],
        )
        try:
            llm.train(
                model=model,
                data=data,
                trainer=trainer,
                log=nemo_logger,
                tokenizer="data",
                optim=opt,
                resume=resume.AutoResume(
                    path=None,  # Overrides the path found by resume_if_exists when set.
                    resume_if_exists=False,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                ),
            )
        except StopAndGoException:
            # Everything is as expected!
            ...

    # Teardown and do it again
    with clean_parallel_state_context():
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_include_optimizer=True,
            find_unused_parameters=True,
        )
        trainer = nl.Trainer(
            devices=1,  ## you can change the numebr of devices to suit your setup
            max_steps=55,
            accelerator="gpu",
            strategy=strategy,
            val_check_interval = 2,
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=False),
            callbacks=[
                TestCheckpointIntegrityCallback(
                    metadata_path=metadata_dir, metadata_keys=["global_step", "learning_rate"]
                ),
                nl_callbacks.ModelCheckpoint(
                    save_best_model=False,
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=50,
                    enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
                    try_restore_best_ckpt=False,
                ),
            ],
        )
        opt_config = OptimizerConfig(
            lr = 1e2,
            optimizer="adam",
            use_distributed_optimizer=True,
        )
        llm.train(
            model=model,
            data=data,
            trainer=trainer,
            log=nemo_logger,
            tokenizer="data",
            optim=opt,
            resume=resume.AutoResume(
                path=None,  # Overrides the path found by resume_if_exists when set.
                resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
            ),
        )


if __name__ == "__main__":
    setup_gpt()
    # test_geneformer_stop_and_go(_geneformer_config())
