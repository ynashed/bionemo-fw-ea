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
import pickle
import torch
from omegaconf.omegaconf import open_dict
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from lightning_lite.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    CheckpointConnector,
)
from pytorch_lightning.callbacks import ModelSummary

from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
)
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager
from nemo.utils.model_utils import import_class_by_path
from nemo.utils.app_state import AppState

try:
    import apex
    from apex.transformer import parallel_state

    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class TrainerBuilder(object):
    @staticmethod
    def adjust_config(cfg):
        if 'global_batch_size' not in cfg.model or \
                cfg.model.get('global_batch_size') is None:
            micro_batch_size = cfg.model.micro_batch_size
            tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
            pipeline_model_parallel_size = cfg.model.pipeline_model_parallel_size
            accumulate_grad_batches = cfg.trainer.get('accumulate_grad_batches', 1)
            n_devices = cfg.trainer.devices * cfg.trainer.num_nodes
            global_batch_size = int(micro_batch_size * n_devices * accumulate_grad_batches / \
                (tensor_model_parallel_size * pipeline_model_parallel_size))

            with open_dict(cfg):
                cfg.model.global_batch_size = global_batch_size
    
    @staticmethod
    def configure_plugins(cfg):
        plugins = []
        if cfg.trainer.precision in [16, "bf16"]:
            scaler = None
            if cfg.trainer.precision == 16:
                scaler = GradScaler(
                    init_scale=cfg.model.get("native_amp_init_scale", 2 ** 32),
                    growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                )
            plugins.append(
                NativeMixedPrecisionPlugin(precision=16, device="cuda", scaler=scaler)
            )

        if cfg.get("cluster_type", None) == "BCP":
            plugins.append(TorchElasticEnvironment())

        return plugins

    @staticmethod
    def configure_callbacks(cfg):
        return [ModelSummary(max_depth=3)]

    @staticmethod
    def configure_strategy(cfg):
        # DDP communication hooks cause errors when used with megatron pipeline parallel
        return NLPDDPStrategy(no_ddp_communication_hook=True)

    @staticmethod
    def resume_checkpoint(cfg, trainer):
        # update resume from checkpoint found by exp_manager
        if cfg.model.resume_from_checkpoint is not None:
            resume_from_checkpoint = cfg.model.resume_from_checkpoint
        else:
            resume_from_checkpoint = (
                trainer._checkpoint_connector.resume_from_checkpoint_fit_path
            )
        logging.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")

        trainer._checkpoint_connector = CheckpointConnector(
            trainer, resume_from_checkpoint=resume_from_checkpoint
        )
        # Override timer callback to a stateless one
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, Timer):
                trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

        # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
        with open_dict(cfg):
            cfg.model.precision = cfg.trainer.precision


class InferenceTrainerBuilder(TrainerBuilder):
    @staticmethod
    def configure_callbacks(cfg):
        return []

    @staticmethod
    def resume_checkpoint(cfg, trainer):
        pass

# FIXME: consolidate this with InferenceTrainerBuilder
class PredictTrainerBuilder(TrainerBuilder):
    @staticmethod
    def configure_callbacks(cfg):
        return []

    @staticmethod
    def resume_checkpoint(cfg, trainer):
        pass

def setup_trainer(cfg, builder=None):
    """NeMo Trainer setup functions"""
    if builder is None:
        builder = TrainerBuilder

    builder.adjust_config(cfg)
    plugins = builder.configure_plugins(cfg)
    callbacks = builder.configure_callbacks(cfg)
    strategy = builder.configure_strategy(cfg)

    trainer = Trainer(
        plugins=plugins, strategy=strategy, callbacks=callbacks, **cfg.trainer
    )
    exp_manager(trainer, cfg.get("exp_manager", None))
    builder.resume_checkpoint(cfg, trainer)
    return trainer


def setup_inference_trainer(cfg, builder=None):
    """NeMo Trainer setup functions for inference"""
    if builder is None:
        builder = InferenceTrainerBuilder

    return setup_trainer(cfg, builder)


def restore_model(restore_path, trainer=None, cfg=None, model_cls=None):
    """Restore model from checkpoint"""
    logging.info(f"Restoring model from {restore_path}")

    # infer model_cls from cfg if missing
    if model_cls is None:
        logging.info(f"Loading model class: {cfg.target}")
        model_cls = import_class_by_path(cfg.target)

    # merge the config from restore_path with the provided config
    restore_cfg = model_cls.restore_from(
        restore_path=restore_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        return_config=True,
    )
    with open_dict(cfg):
        cfg.model = OmegaConf.merge(restore_cfg, cfg.model)

    # build trainer if not provided
    if trainer is None:
        trainer = setup_inference_trainer(cfg=cfg)

    # enforce trainer precition
    with open_dict(cfg):
        cfg.model.precision = trainer.precision

    model = model_cls.restore_from(
        restore_path=restore_path,
        trainer=trainer,
        override_config_path=cfg,
        save_restore_connector=NLPSaveRestoreConnector(),
    )

    return model


# TODO this is taken and from NeMo and we should try to make sure we get this
# back upstream into NeMo
def extract_consumed_samples_from_ckpt(ckpt_path):
    try:
        init_consumed_samples = int(
            float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0])
        )
    except (ValueError, TypeError, IndexError):
        logging.warning(
            "Cannot parse the checkpoint file to get the consumed samples. assume it is zero."
        )
        init_consumed_samples = 0

    return init_consumed_samples


# TODO this is taken and from NeMo and we should try to make sure we get this
# back upstream into NeMo
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


def _reconfigure_inference_batch(global_batch_per_gpu):
    """Reconfigure microbatch sizes for inference."""

    # This should happen only on the last batch of the validation/test dataset with drop_last=False.
    # apex.transformer.pipeline_parallel.utils.get_current_global_batch_size()
    cur_global_batch = (
        apex.transformer.pipeline_parallel.utils.get_current_global_batch_size()
    )
    cur_data_parallel_world_size = parallel_state.get_data_parallel_world_size()
    if global_batch_per_gpu != (cur_global_batch // cur_data_parallel_world_size):
        _reconfigure_microbatch_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=global_batch_per_gpu
            * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_per_gpu,
            data_parallel_size=cur_data_parallel_world_size,
        )

# FIXME: move this to NeMo
def gather_objects(partial_results_list, main_rank=None):
    """
    Collect results from all GPUs.
    Useful for inference over multiple GPUs with DDP.
    
    Args:
        partial_results_list: list of partial results from each GPU
        main_rank: rank of the main process to collect results from all GPUs (useful for collecting results in a target rank)
    """
    # do not fail when DDP is not initialized
    if parallel_state.is_unitialized():
        return partial_results_list
    
    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    # return input when no DDP is used
    if world_size == 1:
        return partial_results_list

    gathered_results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_results, partial_results_list)

    # return None to non-main ranks
    if main_rank is not None:
        if rank != main_rank:
            return None

    # return collected results
    results_list = []
    for r in gathered_results:
        results_list.extend(r)

    return results_list
