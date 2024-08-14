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
from typing import Type

import pytest
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger, io, resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from bionemo.core.model.config import BionemoTrainableModelConfig
from bionemo.example_model import lightning_basic as lb
from bionemo.llm.lightning import LossLoggingCallback
from bionemo.testing import megatron_parallel_state_utils


def _train_model_get_ckpt(
    name: str, root_dir: Path, data_dir: Path, model_cfg_cls: Type[BionemoTrainableModelConfig], ckpt_path: Path | None
) -> Path:
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_best_model=True,
        save_last=True,
        monitor="val_loss",  # TODO find out how to get val_loss logged and use "val_loss",
        save_top_k=1,
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
    model = lb.LitAutoEncoder(config=model_cfg_cls(initial_weights=ckpt_path_optstr))
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=True,
    )
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        limit_val_batches=5,
        val_check_interval=5,
        max_steps=20,
        num_nodes=1,
        log_every_n_steps=5,
        callbacks=[LossLoggingCallback()],
    )
    data_module = lb.MNISTDataModule(data_dir=str(data_dir))  # Re-use the same data directory
    llm.train(
        model=model,
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
    return ckpt_dirpath


@pytest.mark.needs_gpu
def test_train_mnist_litautoencoder_with_megatron_strategy_single_gpu(tmpdir):
    data_dir: Path = tmpdir / "data"
    data_dir.mkdir()
    with megatron_parallel_state_utils.clean_parallel_state_context():

        def train_model(tmpdir: Path) -> Path:
            # Configure our custom Checkpointer
            name = "test_experiment"
            return _train_model_get_ckpt(name, tmpdir, data_dir, lb.ExampleConfig, None)

        ckpt_path = train_model(tmpdir / "pretrain")
        assert ckpt_path.exists()
        assert ckpt_path.is_dir()
        assert io.is_distributed_ckpt(ckpt_path)
    with megatron_parallel_state_utils.clean_parallel_state_context():

        def simple_finetune(tmpdir: Path, ckpt_path: Path) -> Path:
            """Now take that checkpoint and show that you can train it further without making changes to model/loss"""

            # Configure our custom Checkpointer
            name = "simple_finetune_experiment"
            return _train_model_get_ckpt(name, tmpdir, data_dir, lb.ExampleConfig, ckpt_path)

        simple_ft_checkpoint = simple_finetune(tmpdir / "simple_finetune", ckpt_path)
        assert simple_ft_checkpoint.exists()
        assert simple_ft_checkpoint.is_dir()
        assert io.is_distributed_ckpt(simple_ft_checkpoint)
    with megatron_parallel_state_utils.clean_parallel_state_context():

        def add_head_finetune(tmpdir: Path, ckpt_path: Path) -> Path:
            """Now take that checkpoint and show that you can add in a new head/task and update loss"""
            # Configure our custom Checkpointer
            name = "add_head_finetune_experiment"
            return _train_model_get_ckpt(name, tmpdir, data_dir, lb.ExampleFineTuneBothConfig, ckpt_path)

        add_head_checkpoint = add_head_finetune(tmpdir / "add_head_finetune", simple_ft_checkpoint)
        assert add_head_checkpoint.exists()
        assert add_head_checkpoint.is_dir()
        assert io.is_distributed_ckpt(add_head_checkpoint)

    with megatron_parallel_state_utils.clean_parallel_state_context():

        def drop_head_finetune(tmpdir: Path, ckpt_path: Path) -> Path:
            """Now take that checkpoint and show that you can drop a head and update loss"""
            name = "drop_head_finetune_experiment"
            return _train_model_get_ckpt(name, tmpdir, data_dir, lb.ExampleFineTuneDropParentConfig, ckpt_path)

        drop_head_checkpoint = drop_head_finetune(tmpdir / "drop_head_finetune", add_head_checkpoint)
        assert drop_head_checkpoint.exists()
        assert drop_head_checkpoint.is_dir()
        assert io.is_distributed_ckpt(drop_head_checkpoint)
