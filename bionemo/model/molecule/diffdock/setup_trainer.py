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

from typing import Any, Callable

from lightning_fabric.utilities.types import Optimizable
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin, _optimizer_handles_unscaling
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.cuda.amp import GradScaler
from torch.optim.lbfgs import LBFGS

from bionemo.model.utils import (
    TrainerBuilder,
    _reconfigure_microbatch_calculator,
    parallel_state,
    setup_inference_trainer,
)


class CustomMixedPrecisionPlugin(NativeMixedPrecisionPlugin):

    """
    follow this link https://github.com/Lightning-AI/lightning/issues/17407
    modified from https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/plugins/precision/amp.html#MixedPrecisionPlugin.optimizer_step
    """

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: LightningModule,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ):
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(
                optimizer, model=model, optimizer_idx=optimizer_idx, closure=closure, **kwargs
            )
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()

        # EDIT: moved this to the top
        skipped_backward = closure_result is None

        # EDIT: added the second condition
        if not _optimizer_handles_unscaling(optimizer) and not skipped_backward:
            # Unscaling needs to be performed here in case we are going to apply gradient clipping.
            # Optimizers that perform unscaling in their `.step()` method are not supported (e.g., fused Adam).
            # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
            self.scaler.unscale_(optimizer)

        self._after_closure(model, optimizer, optimizer_idx)
        # skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)
            self.scaler.update()
            return step_output
        return closure_result


class DiffdockTrainerBuilder(TrainerBuilder):
    @staticmethod
    def configure_plugins(cfg):
        plugins = []
        if cfg.trainer.precision in [16, "bf16"]:
            scaler = None
            if cfg.trainer.precision == 16:
                scaler = GradScaler(
                    init_scale=cfg.model.get("native_amp_init_scale", 2**32),
                    growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                )
            if cfg.model.get("estimate_memory_usage", None) is None:
                plugins.append(
                    NativeMixedPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler)
                )
            else:
                # with estimate memory usage, in the model forward call during training
                # certain batches may be skipped, which leads to an issue in NativeMixedPrecisionPlugin
                plugins.append(
                    CustomMixedPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler)
                )

        if cfg.get("cluster_type", None) == "BCP":
            plugins.append(TorchElasticEnvironment())

        return plugins

    @staticmethod
    def configure_strategy(cfg):
        return DDPStrategy(find_unused_parameters=True)


class DiffDockModelInference(LightningModule):
    """
    Base class for inference.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = self.load_model(cfg)
        self._trainer = self.model.trainer

    def load_model(self, cfg):
        """Load saved model checkpoint
        Params:
            checkpoint_path: path to nemo checkpoint
        Returns:
            Loaded model
        """
        # load model class from config which is required to load the .nemo file
        model = restore_model(
            restore_path=cfg.restore_from_path,
            cfg=cfg,
        )
        if cfg.get('load_from_checkpoint', None) is not None:
            model.load_from_checkpoint(checkpoint_path=cfg.load_from_checkpoint, strict=False)

        # move self to same device as loaded model
        self.to(model.device)
        # check whether the DDP is initialized
        if parallel_state.is_unitialized():
            logging.info("DDP is not initialized. Initializing...")

            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()
        # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
        _reconfigure_microbatch_calculator(
            rank=0,  # This doesn't matter since it is only used for logging
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
            data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
        )
        model.freeze()
        self.model = model
        return model

    def forward(self, batch):
        """Forward pass of the model"""
        return self.model.model.net(batch)


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
        cfg.model = OmegaConf.merge(restore_cfg, cfg.get('model', OmegaConf.create()))

    # build trainer if not provided
    if trainer is None:
        trainer = setup_inference_trainer(cfg=cfg)
    # enforce trainer precition
    with open_dict(cfg):
        cfg.trainer.precision = trainer.precision

    model = model_cls.restore_from(
        restore_path=restore_path,
        trainer=trainer,
        override_config_path=cfg,
        save_restore_connector=NLPSaveRestoreConnector(),  # SaveRestoreConnector
        strict=False,
    )
    if cfg.get('load_from_checkpoint', None) is not None:
        model.load_from_checkpoint(checkpoint_path=cfg.load_from_checkpoint, strict=False)

    return model
