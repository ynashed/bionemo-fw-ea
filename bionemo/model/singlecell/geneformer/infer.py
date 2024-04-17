# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Any, List, Optional, Tuple

import torch
from torch._tensor import Tensor

from bionemo.model.core.infer import BaseEncoderInference


class GeneformerInference(BaseEncoderInference):
    """
    Initialize the Infer class.

    Args:
        cfg (object): The configuration object.
        model (object, optional): The model object. Defaults to None.
        freeze (bool, optional): Flag to freeze the model. Defaults to True.
        restore_path (str, optional): The path to restore the model from. Defaults to None.
        training (bool, optional): Flag to indicate if the model is in training mode. Defaults to False.
        adjust_config (bool, optional): Flag to adjust the configuration. Defaults to True.
        interactive (bool, optional): Flag to enable interactive mode. Defaults to False.
    """

    def __init__(
        self,
        cfg,
        model=None,
        freeze: bool = True,
        restore_path: Optional[str] = None,
        training: bool = False,
        adjust_config: bool = True,
        interactive: bool = False,
        strict_restore_from_path: bool = True,
    ):
        self.needs_warmup = False  # Doesnt make sense for Geneformer.
        super().__init__(
            cfg=cfg,
            model=model,
            freeze=freeze,
            restore_path=restore_path,
            training=training,
            adjust_config=adjust_config,
            interactive=interactive,
            strict_restore_from_path=strict_restore_from_path,  # NOTE(SKH): This is IMPORTANT, we add tokentype embeddings during fine tuning, if we restore strictly, this will fail.
        )

    def get_example_input_sequence(self) -> List[str]:
        return list(self.tokenizer.vocab.keys())[:64]

    def _tokenize(self, sequences: List[List[str]]) -> List[torch.Tensor]:
        # parent pulls the tokenizer from the loaded model.
        token_ids = [
            torch.tensor(
                [self.tokenizer.token_to_id(gene_symbol) for gene_symbol in gene_symbols],
                device=self.device,
                dtype=torch.long,
            )
            for gene_symbols in sequences
        ]
        return token_ids

    def extract_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids=token_type_ids)
        outputs = outputs[:, 0, :].squeeze()
        return outputs

    def seq_to_hiddens(self, sequences: List[List[str]]) -> Tuple[Tensor, Tensor]:
        '''
        Transforms Sequences into hidden state.
        Should be implemented in a child class, since it is model specific.
        This method returns hidden states and masks.
        Hiddens states contain paddings but do not contain special tokens
        such as <BOS> and <EOS> tokens.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hiddens (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        token_ids, enc_mask = self.tokenize(sequences)
        hiddens = self.model(token_ids, enc_mask, token_type_ids=torch.zeros_like(token_ids))
        if self.model.cfg.get('encoder') is not None and self.model.cfg.encoder.arch == "perceiver":
            # Perceiver outputs a fixed number of embedding tokens (K) regardless of input position.
            #  thus the enc_mask should change to reflect this shape, and the fact that none of the K tokens
            #  should be masked.
            k: int = self.model.cfg.encoder.hidden_steps
            assert hiddens.shape[1] == k
            enc_mask = torch.ones(size=hiddens.shape[:2], dtype=enc_mask.dtype, device=enc_mask.device)

        if hiddens.shape[:2] != enc_mask.shape:
            raise ValueError(
                f"Not using perceiver but got mismatched shapes out of hiddens and enc_mask, please check! {hiddens.shape} vs {enc_mask.shape}"
            )
        return hiddens, enc_mask

    def hiddens_to_embedding(self, hidden_states: Tensor, enc_mask: Tensor) -> Tensor:
        return hidden_states[:, 0, :].squeeze()

    def load_model(
        self, cfg, model: Optional[Any] = None, restore_path: Optional[str] = None, strict: bool = True
    ) -> Any:
        """Load saved model checkpoint. Ensures post_processing is set after the model loads.

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            Loaded model
        """
        # We have to load with strict=False, because our embeddings do not use
        #
        # BioNeMo doesnt expose underlying BERT module to configure forward pass, or component, we cannot allow the pertubation token.
        # on way to handle this token is to re-purpose the token-type embedding in BERT. In the original bert model there are two token types
        #       but we want to set the number of token types to the number of perturbed genes.
        #
        # TODO: set to strict
        model = super().load_model(cfg, model, restore_path, strict=strict)
        model.model.language_model.post_process = cfg.model.get("post_process", False)
        model.model.post_process = cfg.model.get("post_process", False)
        return model
