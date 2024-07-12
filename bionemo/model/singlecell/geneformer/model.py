# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json

from bionemo.core import BioNeMoBertModel
from bionemo.data.singlecell.datamodule import SingleCellDataModule
from bionemo.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.utils.logging import logging


__all__ = [
    "GeneformerModel",
]


class GeneformerModel(BioNeMoBertModel):
    def __init__(self, cfg, trainer, *args, **kwargs):
        # NOTE, we are doing this first to ensure that the tokenizer is loaded before the data module
        self.vocab_file = cfg.tokenizer.vocab_file
        self.medians_file = cfg.data.medians_file
        self._build_tokenizer()  # Need/builds tokenizer
        # Builds dictionary lookup gene_name -> ensembl_id
        self._build_medians(self.medians_file)

        # Uses median dictionary lookup, which is ens -> median. Alos uses gene_to_ens to get median lookup.
        try:
            self.data_module = SingleCellDataModule(
                cfg,
                trainer,
                tokenizer=self.tokenizer,
                median_dict=self.median_dict,
                max_len=cfg.get("seq_length", 2048),
                random_token_prob=cfg.data.get("random_token_prob", 0.1),
                mask_prob=cfg.data.get("mask_prob", 0.15),
                mask_token_prob=cfg.data.get("mask_token_prob", 0.8),
            )
        except Exception as e:
            logging.info(
                f"Failed to build data module: {e}. This is expected at inference/finetune time but not training."
            )
            self.data_module = None

        # this will load the tokenizer again, but doesnt matter :)
        super().__init__(cfg, trainer, *args, **kwargs)

    def _build_tokenizer(self):
        """We keep this method signature fixed because its an unofficial ABC"""
        self.tokenizer = GeneTokenizer.from_vocab_file(
            self.register_artifact("tokenizer.vocab_file", self.vocab_file, verify_src_exists=True)
        )

    def _build_medians(self, medians_file) -> dict:
        """Builds the requisite median dictionaries from a json file via the register_artifact hook"""
        with open(self.register_artifact("data.medians_file", medians_file, verify_src_exists=True), "r") as fp:
            self.median_dict = json.load(fp)
