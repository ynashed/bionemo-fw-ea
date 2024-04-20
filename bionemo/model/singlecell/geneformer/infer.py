# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

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
        inference_batch_size_for_warmup: Optional[int] = None,
    ):
        self.needs_warmup = False  # Not needed for geneformer. See inference test which covers the need for warmup.
        super().__init__(
            cfg=cfg,
            model=model,
            freeze=freeze,
            restore_path=restore_path,
            training=training,
            adjust_config=adjust_config,
            interactive=interactive,
            strict_restore_from_path=strict_restore_from_path,  # NOTE(SKH): This is IMPORTANT, we add tokentype embeddings during fine tuning, if we restore strictly, this will fail.
            inference_batch_size_for_warmup=inference_batch_size_for_warmup,
        )

    def get_example_input_sequence(self) -> List[str]:
        return list(self.tokenizer.vocab.keys())[:64]

    def _tokenize(self, sequences: List[List[str]]) -> List[torch.Tensor]:
        # parent pulls the tokenizer from the loaded model.
        token_ids = [
            torch.tensor(
                [self.tokenizer.class_id] + [self.tokenizer.token_to_id(gene_symbol) for gene_symbol in gene_symbols],
                device=self.device,
                dtype=torch.long,
            )
            for gene_symbols in sequences
        ]
        return token_ids

    def tokens_to_hiddens(self, input_ids, attention_mask, token_type_ids=None, **kwargs) -> torch.Tensor:
        return self.model(input_ids, attention_mask, token_type_ids=token_type_ids, **kwargs)

    def extract_embeddings(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """This function is used for downstream task training where we want to fine-tune the [CLS] token.
            This first calls the forward pass of the bert model on the input_ids, attention_mask, and token_type_ids.
            and then grabs the 0th position (which should be the [CLS] token position) and retuns that as the cell embedding.
            In general you should not use this function, and instead use `seq_to_hiddens` followed by `hiddens_to_embedding` which
            takes care of tokenization for you.

        Args:
            input_ids (_type_): input ids, for example from self.tokenize(...)[0]
            attention_mask (_type_): attention mask, for example also from self.tokenize(...)[1]
            token_type_ids (_type_, optional): If you initialized with `bert_binary_head: True` then you will need to pass in the token_type_ids. Defaults to None.

        Returns:
            _type_: _description_
        """
        with torch.autocast(enabled=self.needs_autocast(), device_type=self.device.type):
            hiddens = self.tokens_to_hiddens(input_ids, attention_mask, token_type_ids=token_type_ids, **kwargs)
        outputs = hiddens[:, 0, :]  # Use the [CLS] token for downstream tasks
        return outputs

    def needs_tokentype_embeddings(self) -> bool:
        """When the model is initialized with `bert_binary_head: True` then it has tokentype_embeddings
        as part of the input/embeddings. This function lets you know if the model was initialized that way.
        Generally we pre-train without this field, but then add it in for fine-tunning tasks downstream, such
        as for PERTURB-seq.
        """
        return self.model.model.language_model.embedding.tokentype_embeddings is not None

    def needs_autocast(self) -> bool:
        return self.model.cfg.precision not in {32, "32"} and self.device.type != "cpu"

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
        if self.needs_tokentype_embeddings():
            token_type_ids = torch.zeros_like(token_ids)
        else:
            token_type_ids = None  # We are just using the pretrained model by itself, and not running inference with any additional token types.
        with torch.autocast(enabled=self.needs_autocast(), device_type=self.device.type):
            hiddens = self.tokens_to_hiddens(token_ids, enc_mask, token_type_ids=token_type_ids)
            if isinstance(hiddens, tuple):
                assert hiddens[1] is None, "We are not expecting a second output from the model."
                hiddens = hiddens[0]

        if hiddens.shape[:2] != enc_mask.shape:
            raise ValueError(
                f"Not using perceiver but got mismatched shapes out of hiddens and enc_mask, please check! {hiddens.shape} vs {enc_mask.shape}"
            )
        return hiddens, enc_mask

    def hiddens_to_embedding(self, hidden_states: Tensor, enc_mask: Tensor) -> Tensor:
        mu_embedding = hidden_states.sum(dim=1) / enc_mask.sum(dim=1, keepdim=True)
        return mu_embedding

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
        # WARNING! if this is true, eg if you do not override pretrain's setting, then you get logits for each position, not the embeddings.
        model.model.language_model.post_process = cfg.model.get("post_process", False)
        model.model.post_process = cfg.model.get("post_process", False)
        return model

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """General forward function for inference on this raw pretrained model. Return the hiddens and embeddings (mean reduced tokens)"""
        if self.needs_tokentype_embeddings():
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(batch["text"]))
        else:
            token_type_ids = None
        with torch.autocast(enabled=self.needs_autocast(), device_type=self.device.type):
            hiddens = self.tokens_to_hiddens(batch["text"], batch["padding_mask"], token_type_ids=token_type_ids)
        embeddings = self.hiddens_to_embedding(hiddens, batch["padding_mask"])
        return {
            "hiddens": hiddens,
            "embeddings": embeddings,
        }
