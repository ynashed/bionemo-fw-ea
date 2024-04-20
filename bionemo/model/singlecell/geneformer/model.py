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


__all__ = [
    "GeneformerModel",
]


class GeneformerModel(BioNeMoBertModel):
    def __init__(self, cfg, trainer, median_dict=None, *args, **kwargs):
        # NOTE, we are doing this first to ensure that the tokenizer is loaded before the data module
        self.vocab_file = cfg.tokenizer.vocab_file
        # self.medians_file = cfg.artifacts.medians_file
        self._build_tokenizer()
        # self.median_dict = self._build_medians(self.medians_file)
        self.median_dict = median_dict
        self.data_module = SingleCellDataModule(
            cfg,
            trainer,
            tokenizer=self.tokenizer,
            median_dict=median_dict,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            max_len=cfg.data.seq_length,
            probabilistic_dirichlet_sampling_train=cfg.data.get("probabilistic_dirichlet_sampling_train", False),
            dirichlet_alpha=cfg.data.get('dirichlet_alpha', 0.5),
        )
        # this will load the tokenizer again, but doesnt matter :)
        super().__init__(cfg, trainer, *args, **kwargs)

    def _build_tokenizer(self):
        '''We keep this method signature fixed because its an unofficial ABC'''
        self.tokenizer = GeneTokenizer.from_vocab_file(self.register_artifact("tokenizer.vocab_file", self.vocab_file))

    def _build_medians(self, medians_file) -> dict:
        '''Builds the requisite median dictionaries from a json file via the register_artifact hook'''
        with open(self.register_artifact("artifacts.medians_file", medians_file), 'r') as fp:
            median_dict = json.load(fp)
        return median_dict
