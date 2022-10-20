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

from pytorch_lightning import Trainer
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from bionemo.model.protein.esm1nv import ESM1nvModel
from nemo.collections.nlp.parts.nlp_overrides import (GradScaler,
                                                      NLPDDPPlugin,
                                                      NLPSaveRestoreConnector)

log = logging.getLogger(__name__)


class ESM1nvInference():
    '''
    All inference functions
    '''

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = self.load_model(cfg)
        self.tokenizer = self.model.tokenizer

    def _tokenize(self, sequences: List[str]):
        # Validate input seqs
        valids = [len(s) > self.model.cfg.seq_length - 2 for s in sequences]
        if True in valids:
            raise Exception(f'One or more sequence exceeds max length({self.model.cfg.seq_length - 2}).')

        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]

        # +2 for the terminal tokens
        pad_length = max([len(seq) for seq in token_ids]) + 2
        mask = [([1] * len(seq)) + ([0] * (pad_length - len(seq))) for seq in token_ids]

        token_ids = [torch.tensor([self.tokenizer.bos_id] + s + [self.tokenizer.eos_id]).cuda() for s in token_ids]
        token_ids = torch.nn.utils.rnn.pad_sequence(token_ids,
                                                    batch_first=True,
                                                    padding_value=0.0)

        mask = torch.tensor(mask).half().cuda()
        return token_ids, mask

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
        embedding = self.model(token_ids, mask, None)

        if self.cfg.model.post_process:
            embedding = embedding[0]

        return embedding, mask

    def load_model(self, cfg):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            ESM trained model
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

        trainer = Trainer(plugins=plugins, **self.cfg.trainer)

        model = ESM1nvModel.restore_from(
            restore_path=cfg.model.downstream_task.input_base_model,
            trainer=trainer,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
        model.model.post_process = self.cfg.model.post_process
        model.half()
        model.freeze()
        return model

    def seq_to_embedding(self, sequences):
        """Compute hidden-state and padding mask for sequences.

        Params
            sequences: strings, input sequences

        Returns
            embedding array and boolean mask
        """

        hiddens, enc_masks = self._transform(sequences)
                # compute average on active hiddens
        lengths = enc_masks.sum(dim=1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError("Empty input is not supported (no token was proveded in one or more of the inputs)")

        embeddings = torch.sum(hiddens*enc_masks.unsqueeze(-1), dim=1) / lengths

        return embeddings
