# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import math
from nemo.utils import logging

from nemo.collections.nlp.parts.nlp_overrides import (
    NLPSaveRestoreConnector,
)

__all__ = ['BioNeMoSaveRestoreConnector']


class BioNeMoSaveRestoreConnector(NLPSaveRestoreConnector):
    """
    SaveRestoreConnector that allows changes in the embedding matrix.
    Will truncate the embedding matrix to vocab_size 
    if the checkpoint has larger dictionary than the model,
    and upsample the embedding matrix if the the checkpoint
    has smaller dictionary.
    """
    # 128 -- is the number of padded vocabulary in MegatronT5Model
    def __init__(self, vocab_size=None) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def modify_state_dict(self, conf, state_dict):
        new_state_dict = {}
        # trunace the word_embeddings and tokens_head
        for key in state_dict.keys():
            if ("word_embeddings" in key) or ("tokens_head" in key):
                # initialize with pretrained word embeddings 
                token_embeddings = state_dict[key]
                if self.vocab_size is None:
                    self.vocab_size = token_embeddings.shape[0]
                logging.info(f"Updating key={key}, token_embeddings.shape={token_embeddings.shape}, vocab_size={self.vocab_size}")
                # tile token_embeddings to be at least self.vocab_size
                dims = (math.ceil(self.vocab_size / token_embeddings.shape[0]),)
                # we want to tile only first dimension
                if len(token_embeddings.shape) == 2:
                    dims += (1,)
                token_embeddings = token_embeddings.tile(dims=dims)
                new_state_dict[key] = token_embeddings[:self.vocab_size]
            elif key.endswith("embedding.position_embeddings.weight"):
                position_embeddings = state_dict[key]
                # allow changing the position embeddings for learned_abs
                if ("encoder_embedding" in key) or ("language_model.embedding" in key):
                    if "encoder" in conf:
                        max_position_embeddings = conf.encoder.max_position_embeddings
                    else:
                        max_position_embeddings = conf.max_position_embeddings
                else:
                    if "decoder" in conf:
                        max_position_embeddings = conf.decoder.max_position_embeddings
                    else:
                        max_position_embeddings = conf.max_position_embeddings
                logging.info(f"Updating key={key}, position_embeddings.shape={position_embeddings.shape}, max_position_embeddings={max_position_embeddings}")
                # tile position_embeddings to be at least max_position_embeddings
                position_embeddings = position_embeddings.tile((math.ceil(max_position_embeddings / position_embeddings.shape[0]), 1))
                new_state_dict[key] = position_embeddings[:max_position_embeddings]
            else:
                new_state_dict[key] = state_dict[key]

        state_dict = new_state_dict
        return state_dict