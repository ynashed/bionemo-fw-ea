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

import logging
import torch

from typing import List

from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.utils.app_state import AppState
from bionemo.model.protein.esm1nv.infer import ESM1nvInference
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from bionemo.model.protein.prott5nv import ProtT5nvModel
from nemo.collections.nlp.parts.nlp_overrides import (GradScaler,
                                                      NLPDDPPlugin,
                                                      NLPSaveRestoreConnector)

log = logging.getLogger(__name__)


class ProtT5nvInference(ESM1nvInference):
    '''
    All inference functions
    '''

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        self.model = self.load_model(cfg)
        self.tokenizer = self.model.tokenizer

    def _transform(self, sequences):
        '''
        Transforms Protein Sequences into hidden state.

        Args:
            sequences (list[str]): list of protein sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        token_ids, mask = self._tokenize(sequences)
        embedding = self.model.encode(tokens_enc=token_ids, enc_mask=mask)

        return embedding, mask

    def load_model(self, cfg):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            ProtT5 trained model
        """
        torch.set_grad_enabled(False)

        plugins = [NLPDDPPlugin()]
        if self.cfg.trainer.precision in [16, 'bf16']:
            scaler = None
            if self.cfg.trainer.precision == 16:
                scaler = GradScaler(
                    init_scale = self.cfg.model.get('native_amp_init_scale', 2 ** 32),
                    growth_interval = self.cfg.model.get('native_amp_growth_interval', 1000),
                )
            plugins.append(NativeMixedPrecisionPlugin(precision=16,
                                                      device='cuda',
                                                      scaler=scaler))


        trainer = Trainer(plugins=plugins, **cfg.trainer)
        assert (
            self.cfg.trainer.devices * self.cfg.trainer.num_nodes
            == self.cfg.model.tensor_model_parallel_size * self.cfg.model.pipeline_model_parallel_size
        ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

        model = ProtT5nvModel.restore_from(
            restore_path=self.cfg.model.downstream_task.input_base_model,
            trainer=trainer,
            save_restore_connector=NLPSaveRestoreConnector(),
        )

        model.freeze()
        return model
