# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

from nemo.utils.model_utils import import_class_by_path
from omegaconf import DictConfig
from pytorch_lightning import Callback

from bionemo.callbacks.logging_callbacks import PerfLoggingCallback, SaveTrainerFinalMetricCallback
from bionemo.utils.dllogger import DLLogger


__all__: Sequence[str] = ("add_test_callbacks", "add_training_callbacks", "setup_dwnstr_task_validation_callbacks")


def add_test_callbacks(cfg: DictConfig, callbacks: List[Callback], mode: str = "train"):
    """
    Appends testing-related callbacks to the list with PL Trainer's callbacks
    Args:
        cfg: config
        mode: str attached to the performance logs
    """

    if cfg.get("create_dllogger_callbacks", False):
        if 'batch_size' in cfg.model.data and cfg.model.data.batch_size != cfg.model.micro_batch_size:
            raise ValueError("cfg.model.data.batch_size should be equal to cfg.model.micro_batch_size")
        dllogger_callbacks_kwargs = cfg.get("dllogger_callbacks_kwargs", {})
        json_file = dllogger_callbacks_kwargs.get("json_file", "dllogger.json")
        append_to_json = dllogger_callbacks_kwargs.get("append_to_json", True)
        use_existing_dllogger = dllogger_callbacks_kwargs.get("use_existing_dllogger", False)
        warmup = dllogger_callbacks_kwargs.get("warmup", 0)
        dllogger = DLLogger(
            json_file=json_file, append_to_json=append_to_json, use_existing_dllogger=use_existing_dllogger
        )
        callbacks.append(
            PerfLoggingCallback(
                dllogger=dllogger, global_batch_size=cfg.model.global_batch_size, mode=mode, warmup=warmup
            )
        )

    if cfg.get("create_trainer_metric_callback", False):
        trainer_metric_callback_kwargs = cfg.get("trainer_metric_callback_kwargs", {})
        callbacks.append(SaveTrainerFinalMetricCallback(**trainer_metric_callback_kwargs))


def _select_dwnstr_task_validation_callbacks(cfg: DictConfig) -> List[DictConfig]:
    """
    Selects configuration of validation callbacks from a main training config

    Params
        cfg: a main configuration of training used in training scripts

    Returns: List of selected validation callback config dicts
    """
    assert "model" in cfg, " The 'model' key is not present in the supplied cfg"
    valid_cbs = []
    if 'dwnstr_task_validation' in cfg.model and cfg.model.dwnstr_task_validation['enabled']:
        valid_cbs = [cfg.model.dwnstr_task_validation.dataset]

    return valid_cbs


def setup_dwnstr_task_validation_callbacks(cfg: DictConfig, plugins: Optional[List[Any]] = None) -> List[Callback]:
    """
    Sets up callbacks for short downstream tasks fine-tunings at the end of the main training validation loop.
    The configuration of callbacks is taken from the main training config.

    Params
        cfg: Dict
        plugins: Optional plugins to be passed to callbacks

    Returns
        List of callbacks to be passed into plt.Trainer
    """
    callbacks_cfg = []
    callbacks_cfg.extend(_select_dwnstr_task_validation_callbacks(cfg))

    callbacks = [
        import_class_by_path(callback_cfg['class'])(callback_cfg, cfg, plugins) for callback_cfg in callbacks_cfg
    ]
    return callbacks


def add_training_callbacks(cfg: DictConfig, callbacks: List[Callback]) -> List[Callback]:
    """
    Sets up callbacks for training loop.
    The configuration of callbacks is taken from the main training config.

    Params
        cfg: Dict
        plugins: Optional plugins to be passed to callbacks

    """
    if 'model' in cfg and 'training_callbacks' in cfg.model:
        callbacks_cfg: List[Dict[str, Any]] = cfg.model.training_callbacks
        for callback_cfg in callbacks_cfg:
            callback_kwargs = dict(deepcopy(callback_cfg))
            callback_cls = callback_kwargs.pop('class')
            callbacks.append(import_class_by_path(callback_cls)(**callback_kwargs))
    return callbacks
