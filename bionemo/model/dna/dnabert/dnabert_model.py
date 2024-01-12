# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from bionemo.core import BioNeMoBertModel
from bionemo.data.fasta_dataset import (
    DNABERTDataModule,
    KmerTokenizerAdapter,
    tokenizers,
)


__all__ = [
    "DNABERTModel",
]


def dna_build_tokenizer(model):
    """
    Default tokenizer is based on available nemo tokenizers.
    Override this method to use an external tokenizer.
    All tokenizers are expected to provide compatible interface.
    """
    tokenizer_type = model._cfg.tokenizer.type
    if tokenizer_type in tokenizers:
        tokenizer = KmerTokenizerAdapter(
            tokenizers[tokenizer_type].from_vocab_file(
                model.register_artifact(
                    "tokenizer.model",
                    model._cfg.tokenizer.model,
                ),
                model.register_artifact(
                    "tokenizer.vocab_file",
                    model._cfg.tokenizer.vocab_file,
                ),
            )
        )
    else:
        tokenizer = get_nmt_tokenizer(
            library=model._cfg.tokenizer.library,
            model_name=tokenizer_type,
            tokenizer_model=model.register_artifact("tokenizer.model", model._cfg.tokenizer.model),
            vocab_file=model.register_artifact("tokenizer.vocab_file", model._cfg.tokenizer.vocab_file),
            legacy=False,
        )
    return tokenizer


class DNABERTModel(BioNeMoBertModel):
    """
    Class for DNABERT models

    """

    def __init__(self, cfg, trainer, *args, **kwargs):
        self.data_module = DNABERTDataModule(cfg, trainer)
        super().__init__(cfg, trainer, *args, **kwargs)

    def _build_tokenizer(self):
        self.tokenizer = dna_build_tokenizer(self)
