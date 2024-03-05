# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict

# Disable logging of invalid SMILES moloecules
from rdkit import RDLogger

from bionemo.data.molecule import MoleculeEnumeration
from bionemo.data.molecule.megamolbart_utils import megamolbart_build_train_valid_test_datasets
from bionemo.model.molecule.mol_enc_dec_model_base import MolEncDecModelBase


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

__all__ = ["MolMIMModel"]


class MolMIMModel(MolEncDecModelBase):
    """
    MolMIM pretraining
    """

    def _setup_collate(self, pad_size_divisible_by_8: bool):
        """
        Sets up collate fn that is required by dataloader used to pretrain MegaMolBART
        Args:
            pad_size_divisible_by_8: should torch.Tensors be padded to sizes divisible by 8?
        Returns:
            callable collate fn
        """
        return MoleculeEnumeration(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            **self._cfg.data,
        ).collate_fn

    def _load_train_valid_test_datasets(self, train_valid_test_num_samples: Dict[str, int]):
        """
        Helper method that sets instance variables corresponding to train, val and test datasets
        Args:
            train_valid_test_num_samples: dicts with number of samples needed for train, val and test steps
        """
        self._train_ds, self._validation_ds, self._test_ds = megamolbart_build_train_valid_test_datasets(
            self._cfg.data, train_valid_test_num_samples
        )
