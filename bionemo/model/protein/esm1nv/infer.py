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

import torch
from typing import List

from bionemo.model.core.infer import BaseEncoderDecoderInference

class ESM1nvInference(BaseEncoderDecoderInference):
    '''
    All inference functions
    '''

    def __init__(self, cfg, model=None):
        super().__init__(cfg=cfg, model=model)
        
    def _tokenize(self, sequences: List[str]):
        """
        ESM expects input format:
        
        encoder input ids - <BOS> + [tokens] + <EOS>
        """
        # Tokenize sequences and add <BOS> and <EOS> tokens
        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]
        token_ids = [torch.tensor([self.tokenizer.bos_id] + s + [self.tokenizer.eos_id]).cuda() for s in token_ids]

        return token_ids

    def seq_to_hiddens(self, sequences):
        '''
        Transforms Sequences into hidden state.
        This class should be implemented in a child class, since it is model specific.
        This class should return only the hidden states, without the special tokens such as
         <BOS> and <EOS> tokens, for example.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        token_ids, enc_mask = self.tokenize(sequences)
        hidden_states = self.model(token_ids, enc_mask, None)

        # ignore <BOS> and <EOS> tokens
        enc_mask[:, 0:2] = 0
        enc_mask = torch.roll(enc_mask, shifts=-1, dims=1)

        # Want to check actual value in the model and not in the config
        if self.model.model.post_process:
            hidden_states = hidden_states[0]

        return hidden_states, enc_mask

    def load_model(self, cfg, model=None):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            ESM trained model
        """
        model = super().load_model(cfg, model=model)
        # control post-processing
        # model.model.post_process = cfg.model.post_process

        return model
