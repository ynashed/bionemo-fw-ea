# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import List, Optional, Sequence, Tuple

import torch

from bionemo.model.core.infer import BaseEncoderInference


__all__: Sequence[str] = ("ESM1nvInference",)


class ESM1nvInference(BaseEncoderInference):
    '''
    All inference functions
    '''

    def __init__(
        self,
        cfg,
        model=None,
        freeze: bool = True,
        restore_path: Optional[str] = None,
        training: bool = False,
        adjust_config: bool = True,
        interactive: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            model=model,
            freeze=freeze,
            restore_path=restore_path,
            training=training,
            adjust_config=adjust_config,
            interactive=interactive,
        )

    def _tokenize(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        ESM expects input format:

        encoder input ids - <BOS> + [tokens] + <EOS>
        """
        # Tokenize sequences and add <BOS> and <EOS> tokens
        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]
        token_ids = [torch.tensor([self.tokenizer.bos_id] + s + [self.tokenizer.eos_id]).cuda() for s in token_ids]

        return token_ids

    def seq_to_hiddens(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Transforms Sequences into hidden state.
        Should be implemented in a child class, since it is model specific.
        This method returns hidden states and masks.
        Hidden states are returned for all tokens, including <BOS>, <EOS> and padding.
        <BOS>, <EOS> and padding are masked out.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for special tokens (<BOS> and <EOS>) and padded sections
        '''
        token_ids, enc_mask = self.tokenize(sequences)
        hidden_states = self.model.encode(token_ids, enc_mask, reconfigure_microbatch=not self.interactive)

        # ignore <BOS> and <EOS> tokens
        enc_mask[:, 0:2] = 0
        enc_mask = torch.roll(enc_mask, shifts=-1, dims=1)

        return hidden_states, enc_mask

    def load_model(self, cfg, model=None, restore_path=None):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            ESM trained model
        """
        # control post-processing
        if model is None:
            post_process = cfg.model.post_process
        else:
            post_process = model.model.post_process
        model = super().load_model(cfg, model=model, restore_path=restore_path)

        model.model.post_process = post_process

        return model
