# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import List, Optional

import torch
from torch.cuda.amp import autocast

from bionemo.model.core.infer import BaseEncoderDecoderInference


class DNABERTInference(BaseEncoderDecoderInference):
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

    def _tokenize(self, sequences: List[str]):
        # parent pulls the tokenizer from the loaded model.
        token_ids = [self.model.tokenizer.text_to_ids(s) for s in sequences]

        return token_ids

    def tokenize(self, sequences: List[str]):
        '''
        Note that the parent includes padding, likely not necessary for DNABERT.
        This actually fails if you let it call the parent super class, since it expects padding to be a thing.
        '''
        return self._tokenize(sequences)

    def seq_to_hiddens(self, sequences: List[str]):
        token_ids = torch.tensor(self.tokenize(sequences), device=self.model.device)
        padding_mask = torch.ones(size=token_ids.size(), device=self.model.device)

        with autocast(enabled=True):
            output_tensor = self.model(token_ids, padding_mask, token_type_ids=None, lm_labels=None)

        # Padding mask gets used for automatically adjusting the length of the sequence with respect to padding tokens.
        #       DNABERT does not have a padding token, so this is redundant.
        return output_tensor, padding_mask

    def load_model(self, cfg, model=None, restore_path=None):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            ESM trained model
        """
        # control post-processing, load_model messes with our config so we need to bring it in here.
        model = super().load_model(cfg, model=model, restore_path=restore_path)
        # Hardcoded for DNABERT as there is no post-processing
        model.model.post_process = False
        return model
