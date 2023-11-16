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


from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies.ddp import DDPStrategy

from bionemo.model.utils import (
    TrainerBuilder,
    _reconfigure_microbatch_calculator,
    parallel_state,
    setup_inference_trainer,
)


class DiffdockTrainerBuilder(TrainerBuilder):
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
