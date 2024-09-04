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


from pathlib import Path
from typing import Any, Dict, Set, Tuple, Type

import pytest
import torch
from _pytest.compat import LEGACY_PATH
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger, io, resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.example_model import lightning_basic as lb
from bionemo.llm.lightning import LossLoggingCallback
from bionemo.llm.model.config import MegatronBioNeMoTrainableModelConfig
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.callbacks import MetricTracker
from bionemo.testing.data.load import BIONEMO_CACHE_DIR


def _train_model_get_ckpt(
    name: str,
    root_dir: Path,
    model_cfg_cls: Type[MegatronBioNeMoTrainableModelConfig],
    ckpt_path: Path | None,
    skip_weight_prefixes: Set[str],
    precision: PrecisionTypes,
) -> Tuple[Path, MetricTracker]:
    if precision not in {"32", 32}:
        extra_args: Dict[str, Any] = {
            "plugins": nl.MegatronMixedPrecision(precision=precision),
        }
    else:
        extra_args = {}

    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_best_model=False,
        save_last=True,
        save_on_train_epoch_end=True,
        monitor="reduced_train_loss",  # TODO find out how to get val_loss logged and use "val_loss",
        every_n_train_steps=5,
        enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
        # async_save=False,  # Tries to save asynchronously, previously led to race conditions.
    )
    save_dir = root_dir / name
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)
    # Setup the logger and train the model
    nemo_logger = NeMoLogger(
        dir=str(root_dir),
        name=name,
        tensorboard=tb_logger,
        ckpt=checkpoint_callback,
    )
    # Needed so that the trainer can find an output directory for the profiler
    # nemo_logger.save_dir = tmpdir
    # ckpt_path needs to be a string for SerDe
    ckpt_path_optstr: str | None = str(ckpt_path) if ckpt_path is not None else None
    config = model_cfg_cls(
        initial_ckpt_path=ckpt_path_optstr,
        initial_ckpt_skip_keys_with_these_prefixes=sorted(skip_weight_prefixes),
        # NOTE: the optimizer needs fp16 and bf16 bools set to match the model. For now get them from the config and
        #  set them here.
        fp16=get_autocast_dtype(precision) == torch.float16,
        bf16=get_autocast_dtype(precision) == torch.bfloat16,
        autocast_dtype=get_autocast_dtype(precision),
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        enable_autocast=precision not in {32, "32"},
    )

    lightning_module = lb.LitAutoEncoder(
        config=config,
    )
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=True,
    )
    metric_tracker = MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"])
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        limit_val_batches=5,
        val_check_interval=5,
        max_steps=100,
        num_nodes=1,
        log_every_n_steps=5,
        callbacks=[LossLoggingCallback(), metric_tracker],
        **extra_args,
    )
    data_module = lb.MNISTDataModule(data_dir=str(BIONEMO_CACHE_DIR), batch_size=64)  # Re-use the same data directory
    llm.train(
        model=lightning_module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            path=None,  # Overrides the path found by resume_if_exists when set.
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
    ckpt_dirpath = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_dirpath, metric_tracker


@pytest.mark.needs_gpu
@pytest.mark.parametrize("precision", [32, "bf16-mixed"])
def test_train_mnist_litautoencoder_with_megatron_strategy_single_gpu(tmpdir: LEGACY_PATH, precision: PrecisionTypes):
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        ckpt_path, initial_metrics = _train_model_get_ckpt(
            name="test_experiment",
            root_dir=tmpdir / "pretrain",
            model_cfg_cls=lb.ExampleConfig,
            ckpt_path=None,
            skip_weight_prefixes=set(),
            precision=precision,
        )
        assert ckpt_path.exists()
        assert ckpt_path.is_dir()
        assert io.is_distributed_ckpt(ckpt_path)
        assert initial_metrics.collection_train["loss"][0] > initial_metrics.collection_train["loss"][-1]
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        simple_ft_checkpoint, simple_ft_metrics = _train_model_get_ckpt(
            name="simple_finetune_experiment",
            root_dir=tmpdir / "simple_finetune",  # new checkpoint will land in a subdir of this
            model_cfg_cls=lb.ExampleConfig,  # same config as before since we are just continuing training
            ckpt_path=ckpt_path,  # specify the initial checkpoint path now
            skip_weight_prefixes=set(),  # no new weights in this model need skipping
            precision=precision,
        )
        assert simple_ft_checkpoint.exists()
        assert simple_ft_checkpoint.is_dir()
        assert io.is_distributed_ckpt(simple_ft_checkpoint)
        assert initial_metrics.collection_train["loss"][-1] > simple_ft_metrics.collection_train["loss"][0]
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        add_head_checkpoint, add_head_ft_metrics = _train_model_get_ckpt(
            name="add_head_finetune_experiment",
            root_dir=tmpdir / "add_head_finetune",
            model_cfg_cls=lb.ExampleFineTuneBothConfig,  # config that returns a model/loss with a new task head
            ckpt_path=simple_ft_checkpoint,  # cumulatively modify a checkpoint with subsequent experiments, (optional)
            skip_weight_prefixes={"digit_classifier"},  # The new head weights are not in the ckpt so need skipping.
            precision=precision,
        )
        assert add_head_checkpoint.exists()
        assert add_head_checkpoint.is_dir()
        assert io.is_distributed_ckpt(add_head_checkpoint)
        assert add_head_ft_metrics.collection_train["loss"][0] > add_head_ft_metrics.collection_train["loss"][-1]
        # We're adding a new loss, so the loss should be worse initially at least.
        assert add_head_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

    with megatron_parallel_state_utils.distributed_model_parallel_state():
        drop_head_checkpoint, drop_head_ft_metrics = _train_model_get_ckpt(
            name="drop_head_finetune_experiment",
            root_dir=tmpdir / "drop_head_finetune",
            model_cfg_cls=lb.ExampleFineTuneDropParentConfig,  # config that drops the decoder and head -> only cls now
            ckpt_path=add_head_checkpoint,  # cumulatively build on the config that had this cls head (optional)
            skip_weight_prefixes=set(),  # no new parameters vs prior cfg, will continue training cls head by itself
            precision=precision,
        )
        assert drop_head_checkpoint.exists()
        assert drop_head_checkpoint.is_dir()
        assert io.is_distributed_ckpt(drop_head_checkpoint)
        # We're dropping a loss, so initially we should be better than before
        assert drop_head_ft_metrics.collection_train["loss"][0] > drop_head_ft_metrics.collection_train["loss"][-1]
        assert add_head_ft_metrics.collection_train["loss"][-1] > drop_head_ft_metrics.collection_train["loss"][0]
