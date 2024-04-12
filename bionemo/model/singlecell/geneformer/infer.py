# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Any, List, Optional

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

    def _tokenize(self, gene_symbols: List[str]):
        # parent pulls the tokenizer from the loaded model.
        token_ids = self.tokenizer.tokens_to_id(gene_symbols)
        return token_ids

    def tokenize(self, sequences: List[str]):
        '''Parent implementation adds BOS/EOS tokens, but this model has no BOS/EOS concept'''
        return self._tokenize(sequences)

    def extract_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids=token_type_ids)
        outputs = outputs[:, 0, :].squeeze()
        return outputs

    def load_model(self, cfg, model: Optional[Any] = None, restore_path: Optional[str] = None) -> Any:
        """Load saved model checkpoint. Ensures post_processing is set after the model loads.

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            Loaded model
        """

        model = super().load_model(cfg, model, restore_path)
        model.model.language_model.post_process = cfg.model.get("post_process", False)
        model.model.post_process = cfg.model.get("post_process", False)
        return model
