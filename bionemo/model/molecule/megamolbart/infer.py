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

from bionemo.model.core.infer import BaseEncoderDecoderInference

log = logging.getLogger(__name__)
__all__ = ["MegaMolBARTInference"]

class MegaMolBARTInference(BaseEncoderDecoderInference):
    '''
    All inference functions
    '''

    def __init__(self, cfg, model=None):
        super().__init__(cfg=cfg, model=model)
        
    def _tokenize(self, sequences: List[str]):
        """
        ProtT5 expects input/output format:
        
        encoder input ids - [tokens] (without <BOS> and <EOS>)
        decoder input ids - <BOS> + [tokens]
        decoder output ids - [tokens] + <EOS>
        """
        # Tokenize sequences
        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]

        return token_ids

    def seq_to_hiddens(self, sequences):
        '''
        Transforms Sequences into hidden state.
        Should be implemented in a child class, since it is model specific.
        This method returns hidden states and masks.
        Hiddens states contain paddings but do not contain special tokens 
        such as <BOS> and <EOS> tokens.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        token_ids, enc_mask = self.tokenize(sequences)
        embedding = self.model.encode(tokens_enc=token_ids, enc_mask=enc_mask)

        return embedding, enc_mask

    def hiddens_to_seq(self, hidden_states, enc_mask):
        '''
        Transforms hidden state into sequences (i.e., sampling in most cases).
        This class should be implemented in a child class, since it is model specific.
        This class should return the sequence with special tokens such as
         <BOS> and <EOS> tokens, if used.

        Args:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            sequences (list[str]): list of sequences
        '''
        predicted_tokens_ids, _ = self.model.decode(tokens_enc=None,
                                                    enc_mask=enc_mask,
                                                    num_tokens_to_generate=self.model.cfg.max_position_embeddings,
                                                    enc_output=hidden_states)
        sequences = self.detokenize(tokens_ids=predicted_tokens_ids)
        
        return sequences

    @property
    def default_sampling_kwargs(self):
        """
        Returns a dict of default sampling kwargs per sampling method.
        Example:
            {
                "greedy-perturbate": {"scaled_radius": 1, "topk": 10},
                "beam-search": {"beam_size": 5, "beam_alpha": 0.6, "beam_min_length": 1, "beam_max_length": 100},
            }
            
        Should be overridden in child class if sampling is supported.
        """
        return {
                # smis - a list of SMILES strings to perturbate num_samples times each
                "greedy-perturbate": {"scaled_radius": 1, "topk": 10, "smis": []},
            }
        
    def sample(self,
               num_samples=10,
               return_embedding=False,
               sampling_method="greedy-perturbate",
               **sampling_kwarg):
        """
        Sample from the model given sampling_method.
        
        Args:
            num_samples (int): number of samples to generate (depends on sampling method)
            return_embedding (bool): return embeddings corresponding to each of the samples in addition to the samples
            sampling_method (str): sampling method to use. Should be replaced with default sampling method in child class
            sampling_kwarg (dict): kwargs for sampling method. Depends on the sampling method.
        """
        # get sampling kwargs
        default_sampling_kwarg = self.default_sampling_kwargs
        if not sampling_method in default_sampling_kwarg:
            raise ValueError(f'Invalid samping method {sampling_method}, supported sampling methods are {default_sampling_kwarg.keys()}')
        
        cur_sampling_kwarg = default_sampling_kwarg[sampling_method].copy()
        cur_sampling_kwarg.update(sampling_kwarg)
        sampling_kwarg = cur_sampling_kwarg
        
        # execute selected sampling method
        if sampling_method == 'greedy-perturbate':
            smis = sampling_kwarg["smis"]
            if not len(smis):
                raise ValueError(f'No SMILES strings provided for sampling via "smis" argument')
            
            hidden_states, enc_masks = self.seq_to_hiddens(smis)
            scaled_radius = sampling_kwarg['scaled_radius']
            sample_masks = enc_masks.repeat_interleave(num_samples, 0)
            perturbed_hiddens = hidden_states.repeat_interleave(num_samples, 0)
            perturbed_hiddens = perturbed_hiddens + (scaled_radius * torch.randn(perturbed_hiddens.shape).to(perturbed_hiddens.device))

            samples = self.hiddens_to_seq(perturbed_hiddens, sample_masks)
            if return_embedding:
                embs = self.hiddens_to_embedding(perturbed_hiddens, sample_masks)

        if return_embedding:
            return samples, embs
        else:
            return samples