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

from typing import List, Optional

from nemo.utils.model_utils import import_class_by_path
from omegaconf import DictConfig


__all__ = ["setup_callbacks"]


def _select_dwnstr_task_validation_callbacks(cfg: DictConfig) -> List:
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


def setup_callbacks(cfg, plugins: Optional[List] = None) -> List:
    """
    Sets up callbacks for short downstream tasks fine-tunings at the end of the main training validation loop.
    The configuration of callbacks is taken from the main training config.

    Params
        cfg: Dict
        plugins: Optional plugins to be passed to callbacks

    Returns
        List of callbacks to be passed into plt.Trainer
    TODO(dorotat): This method can be generalized to instantiate any method from config, ie using hydra instantiate or defined specific layout of input config

    """
    callbacks_cfg = []
    callbacks_cfg.extend(_select_dwnstr_task_validation_callbacks(cfg))

    callbacks = [
        import_class_by_path(callback_cfg['class'])(callback_cfg, cfg, plugins) for callback_cfg in callbacks_cfg
    ]
    return callbacks
