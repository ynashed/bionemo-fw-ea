# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os
import pickle

from bionemo.core import BioNeMoBertModel
from bionemo.data.singlecell.datamodule import SingleCellDataModule
from bionemo.tokenizer.gene_tokenizer import GeneTokenizer


__all__ = [
    "GeneformerModel",
]


class GeneformerModel(BioNeMoBertModel):
    def __init__(self, cfg, trainer, tokenizer=None, median_dict=None, *args, **kwargs):
        # NOTE: we do all this dumb delayed stateful-shit because we want to setup the data module, but we dont want to instantiate the data yet.
        self.data_module = SingleCellDataModule(
            cfg,
            trainer,
            tokenizer=tokenizer,
            median_dict=median_dict,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
        )
        super().__init__(cfg, trainer, *args, **kwargs)

    def _build_tokenizer(self):
        tokenizer_path = os.path.join(self.cfg.data.dataset_path, "gene_name_id_dict.pkl")
        self.tokenizer = geneformer_build_tokenizer(tokenizer_path)


def geneformer_build_tokenizer(tokenizer_path: str) -> GeneTokenizer:
    """
    Loads the tokenizer from the pre-defined dataset path and downloaded pickle object.

    NOTE: this is not a good approach, but lets us use the -exact- same setup as GeneFormer
    """
    # gene_ens = pickle.load(open(os.path.join(model.cfg.data.dataset_path, "gene_name_id_dict.pkl"), 'rb'))
    with open(tokenizer_path, 'rb') as f:
        gene_ens = pickle.load(f)
    tokenizer = GeneTokenizer()
    tokenizer.build_vocab(gene_ens)
    return tokenizer
