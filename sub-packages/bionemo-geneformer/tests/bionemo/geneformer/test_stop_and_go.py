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
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.testing.megatron_parallel_state_utils import (
    clean_parallel_state_context,
    distributed_model_parallel_state,
)
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
        global_batch_size=2 * int(devices),  # micro batch size times divices
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )
    return data


# TODO (@skothenhill) How can we adapt this into a test harness?
def test_geneformer_stop_and_go(seed: int = 42):
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

    def setup_geneformer_model():
        # TODO:
        return model, data, opt

    def setup_trainer_and_strategy_geneformer(mode: Literal["stop", "go"], metrics=[]):
        # TODO:
        return trainer
    

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
    with distributed_model_parallel_state():
        module = BioBertLightningModule(config=_geneformer_config(), tokenizer=tokenizer, optimizer=optim)
        tp_size = 1
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            ddp="megatron",
            find_unused_parameters=True,
            ckpt_include_optimizer=True,
        )

        trainer = nl.Trainer(
            devices=devices,
            max_steps=num_steps,  # Hardcoded to debug
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=2,  # Hardcoded to coyp pretrain
            val_check_interval=val_check_interval,
            num_nodes=1,
            callbacks=[
                io.track_io(MetadataSaveCallback)(
                    # metadata_path=metadata_dir, metadata_keys=["learning_rate", "global_step"]
                    metadata_path=metadata_dir,
                    metadata_keys=["global_step", "val_loss"],
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
    with distributed_model_parallel_state():
        devices = 1
        tp_size = 1
        pp_size = 1
        module = BioBertLightningModule(config=_geneformer_config(), tokenizer=tokenizer, optimizer=optim)
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            ddp="megatron",
            find_unused_parameters=True,
            ckpt_include_optimizer=True,
        )
        data_module = make_real_datamodule(
            tokenizer,
            seq_length,
            median_dict,
            devices=devices,
            tensor_model_parallel_size=tp_size,
            data_path=data_path,
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
                TestCheckpointIntegrityCallback(metadata_path=metadata_dir, metadata_keys=["global_step", "val_loss"]),
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
            optim=optim,
            resume=resume.AutoResume(
                path=None,  # Overrides the path found by resume_if_exists when set.
                resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
            ),
        )


# Here is code that does the same thing for GPT.
def test_gpt_example():
    import torch
    from megatron.core.optimizer import OptimizerConfig
    from nemo import lightning as nl
    from nemo.collections import llm

    ## setup the dummy dataset

    # Setup model, this should be the same for both calls.
    def setup_gpt_model():
        seq_length = 128
        global_batch_size = 16
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
        data = llm.MockDataModule(seq_length=seq_length, global_batch_size=global_batch_size, num_val_samples=32)
        model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)

        lr = 1e-4
        opt_config = OptimizerConfig(
            lr=lr,
            optimizer="adam",
            use_distributed_optimizer=True,
        )
        num_steps = 500
        cosine_rampup_frac = 0.1
        cosine_hold_frac = 0.05
        opt = nl.MegatronOptimizerModule(
            config=opt_config,
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                min_lr=lr / 100,
                warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
                interval="step",
                monitor="reduced_train_loss",
                constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
            ),
        )
        return model, data, opt

    # Setup trainer and strategy- this one is trickier becuase callbacks change depending on context, so realistically
    #       we should just do these live.
    def setup_trainer_and_strategy_gpt(mode: Literal["stop", "go"], metrics=[]):
        if mode == "stop":
            # To stop we must track IO for all callbacks and setup the save and raise callbacks.
            callbacks = [
                io.track_io(MetadataSaveCallback)(
                    metadata_path=metadata_dir,
                    metadata_keys=metrics,
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
            ]
        elif mode == "go":
            # To go we cannot track io, but we must setup the integrity callback.
            callbacks = [
                TestCheckpointIntegrityCallback(metadata_path=metadata_dir, metadata_keys=metrics),
                nl_callbacks.ModelCheckpoint(
                    save_best_model=False,
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=50,
                    enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
                    try_restore_best_ckpt=False,
                ),
            ]
        # Finally we can setup the trainer and strategy.
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_include_optimizer=True,
        )
        trainer = nl.Trainer(
            devices=1,
            max_steps=4,
            accelerator="gpu",
            strategy=strategy,
            val_check_interval=2,
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=False),
            callbacks=callbacks,
        )
        return trainer

    exp_name = "gpt_stop_and_go"
    root_dir = bionemo2_root / "results"
    metadata_dir = root_dir / exp_name
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
        try:
            model, data, opt = setup_gpt_model()
            trainer = setup_trainer_and_strategy_gpt("stop", metrics=["global_step"])
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
        model, data, opt = setup_gpt_model()
        trainer = setup_trainer_and_strategy_gpt("go", metrics=["global_step"])
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
    test_gpt_example()
    # test_geneformer_stop_and_go(_geneformer_config())
