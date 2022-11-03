# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

import re
from omegaconf.omegaconf import open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.callbacks import ModelSummary
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager
from nemo.utils.app_state import AppState


class TrainerBuilder(object):
    @staticmethod
    def configure_plugins(cfg):
        plugins = [
            NLPDDPPlugin(
                no_ddp_communication_hook=True,
                find_unused_parameters=False,
            )
        ]
        if cfg.trainer.precision in [16, 'bf16']:
            scaler = None
            if cfg.trainer.precision == 16:
                scaler = GradScaler(
                    init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                    growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                )
            plugins.append(NativeMixedPrecisionPlugin(precision=16, device='cuda', scaler=scaler))

        if cfg.get('cluster_type', None) == 'BCP':
            plugins.append(TorchElasticEnvironment())

        return plugins

    @staticmethod
    def configure_callbacks(cfg):
        return [ModelSummary(max_depth=3)]

    @staticmethod
    def resume_checkpoint(cfg, trainer):
            # update resume from checkpoint found by exp_manager
        if cfg.model.resume_from_checkpoint is not None:
            resume_from_checkpoint = cfg.model.resume_from_checkpoint
        else:
            resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

        trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
        # Override timer callback to a stateless one
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, Timer):
                trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

        # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
        with open_dict(cfg):
            cfg.model.precision = cfg.trainer.precision


def setup_trainer(cfg, builder=None):
    """NeMo Trainer setup functions"""
    if builder is None:
        builder = TrainerBuilder

    plugins = builder.configure_plugins(cfg)
    callbacks = builder.configure_callbacks(cfg)

    trainer = Trainer(plugins=plugins, **cfg.trainer, callbacks=callbacks)
    exp_manager(trainer, cfg.get("exp_manager", None))
    builder.resume_checkpoint(cfg, trainer)
    return trainer

def extract_consumed_samples_from_ckpt(ckpt_path):
    try:
        init_consumed_samples = int(float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0]))
    except (ValueError, TypeError, IndexError):
        logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
        init_consumed_samples = 0

    return init_consumed_samples

def compute_consumed_samples(model, steps_since_resume=0):
    app_state = AppState()
    consumed_samples = (
        model.init_consumed_samples
        + steps_since_resume
        * app_state.data_parallel_size
        * model.cfg.micro_batch_size
        * model.trainer.accumulate_grad_batches
    )
    return int(consumed_samples)
