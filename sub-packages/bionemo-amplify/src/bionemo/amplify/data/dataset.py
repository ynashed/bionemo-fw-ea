# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from pathlib import Path
from typing import Sequence, TypeVar, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets import load_dataset as hf_load_dataset

from bionemo.amplify.data import tokenizer
from bionemo.llm.data import masking
from bionemo.llm.data.types import BertSample


class AMPLIFYMaskedResidueDataset(Dataset):
    """Dataset class for AMPLIFY pretraining that implements sampling of UR100P sequences.
    """

    def __init__(
        self,
        hf_dataset_name: str | os.PathLike,
        split: Literal["train", "test"] = "train",
        seed: int = np.random.SeedSequence().entropy,  # type: ignore
        max_seq_length: int = 512,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        mask_random_prob: float = 0.1,
        tokenizer: tokenizer.BioNeMoAutoTokenizer = tokenizer.get_tokenizer(),
    ) -> None:
        """Initializes the dataset.

        Args:
            hf_dataset_name: Name of the Hugging Face dataset containing UR100P protein sequences.
            split: The split of the dataset to use ["train", "test"]. Defaults to "train".
            total_samples: Total number of samples to draw from the dataset.
            seed: Random seed for reproducibility. This seed is mixed with the index of the sample to retrieve to ensure
                that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
                generated.
            max_seq_length: Crop long sequences to a maximum of this length, including BOS and EOS tokens.
            mask_prob: The overall probability a token is included in the loss function. Defaults to 0.15.
            mask_token_prob: Proportion of masked tokens that get assigned the <MASK> id. Defaults to 0.8.
            mask_random_prob: Proportion of tokens that get assigned a random natural amino acid. Defaults to 0.1.
            tokenizer: The input AMPLIFY tokenizer. Defaults to the standard AMPLIFY tokenizer.
        """
        self.protein_dataset = hf_load_dataset(hf_dataset_name, data_dir="UniProt", split=split)
        self.total_samples = len(self.protein_dataset)
        self.seed = seed
        self.max_seq_length = max_seq_length

        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer does not have a mask token.")

        self.mask_config = masking.BertMaskConfig(
            tokenizer=tokenizer,
            random_tokens=range(4, 24),
            mask_prob=mask_prob,
            mask_token_prob=mask_token_prob,
            random_token_prob=mask_random_prob,
        )

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the total number of sequences in the dataset.
        """
        return self.total_samples

    def __getitem__(self, idx: int) -> BertSample:
        """Deterministically masks and returns a protein sequence from the dataset.
        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A (possibly-truncated), masked protein sequence with CLS and EOS tokens and associated mask fields.
        """
        if idx not in range(len(self)):
            raise IndexError(f"Index {idx} out of range [0, {len(self)}).")

        # Initialize a random number generator with a seed that is a combination of the dataset seed and the index.
        rng = np.random.default_rng([self.seed, idx])
        sequence = self.protein_dataset[idx]

        # We don't want special tokens before we pass the input to the masking function; we add these in the collate_fn.
        tokenized_sequence = self._tokenize(sequence)
        cropped_sequence = _random_crop(tokenized_sequence, self.max_seq_length, rng)

        torch_seed = rng.integers(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max)
        masked_sequence, labels, loss_mask = masking.apply_bert_pretraining_mask(
            tokenized_sequence=cropped_sequence,  # type: ignore
            random_seed=torch_seed,
            mask_config=self.mask_config,
        )

        return {
            "text": masked_sequence,
            "types": torch.zeros_like(masked_sequence, dtype=torch.int64),
            "attention_mask": torch.ones_like(masked_sequence, dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(masked_sequence, dtype=torch.int64),
        }

    def _tokenize(self, sequence: str) -> torch.Tensor:
        """Tokenize a protein sequence.

        Args:
            sequence: The protein sequence.

        Returns:
            The tokenized sequence.
        """
        tensor = self.tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
        return tensor.flatten()  # type: ignore


_T = TypeVar("_T", str, torch.Tensor)


def _random_crop(s: _T, crop_length: int, rng: np.random.Generator) -> _T:
    """Randomly crops a input to a maximum length."""
    if crop_length >= len(s):
        return s

    start_index = rng.integers(0, len(s) - crop_length)
    return s[start_index : start_index + crop_length]