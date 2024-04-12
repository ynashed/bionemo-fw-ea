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
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
from torch.utils.data import Dataset

from bionemo.data.singlecell.utils import sample_or_truncate_plus_pad
from tokenizers import Tokenizer


class SingleCellDataset(Dataset):
    """
    A dataset class for single-cell pre-training. These can be generated using the sc_memmap.py script. Future
    updates will contain more comprehensive workflows for generating a Sparse Memmap from scRNA-seq.

    Args:
        data_path (str): Path where the single cell files are stored. It should contain the following files:
            - `metadata.json`: Path containing feature subset associated with each dataset.
            - `features.csv`: Feature subset associated with each sample.
            - Gene expression matrix stored in CSR format as `numpy.memmap`:
                - `gene_expression_data.npy`: Gene expression values.
                - `gene_expression_ind.npy`: Gene indices associated with gene values.
                - `gene_expression_ptr.npy`: Column indices for each sample.
        tokenizer: The tokenizer to use for tokenizing the input data.
        median_dict (dict, optional): A dictionary containing median values for each gene. Defaults to None.
        max_len (int, optional): The maximum length of the input sequence. Defaults to 1024.

    Attributes:
        data_path (str): Path where the single cell files are stored.
        max_len (int): The maximum length of the input sequence.
        metadata (dict): Metadata loaded from `metadata.json`.
        gene_medians (dict): A dictionary containing median values for each gene. If None, a median of '1' is assumed for all genes.
        num_train (int): The number of samples in the training split.
        num_val (int): The number of samples in the validation split.
        num_test (int): The number of samples in the test split.
        index_offset (int): The offset to apply to the indices.
        length (int): The total number of samples in the dataset.
        gene_data (numpy.memmap): Gene expression values stored in CSR format.
        gene_data_indices (numpy.memmap): Gene indices associated with gene values.
        gene_data_ptr (numpy.memmap): Column indices for each sample.
        tokenizer: The tokenizer used for tokenizing the input data.
        dataset_ccum (numpy.ndarray): Cumulative sum of row counts to map row indices to dataset id.
        dataset_map (dict): Mapping of dataset id to dataset name.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    See Also:
        bionemo/data/singlecell/sc_memmap.py - creates the artifacts required for instantiating a singlecell dataset from hdf5 files.
    """

    def __init__(self, data_path: str, tokenizer: Any, median_dict: Optional[dict] = None, max_len: int = 1024):
        super().__init__()
        self.data_path = data_path
        self.max_len = max_len
        path = Path(data_path)

        # - metadata
        self.metadata = json.load(open(path / 'metadata.json', 'r'))

        # - median dict
        self.gene_medians = median_dict

        # - train/val idxs sampled contiguously
        total_el = sum([v["num_el"] for _, v in self.metadata.items()])
        self.num_samples = sum([v["shape"][0] for _, v in self.metadata.items()])
        # - load data
        self.gene_data = np.memmap(path / 'gene_expression_data.npy', dtype='float32', mode='r', shape=(total_el,))

        self.gene_data_indices = np.memmap(
            path / 'gene_expression_ind.npy', dtype='int32', mode='r', shape=(total_el,)
        )

        self.gene_data_ptr = np.memmap(
            path / 'gene_expression_ptr.npy', dtype='int64', mode='r', shape=(self.num_samples + 1,)
        )
        self.tokenizer = tokenizer

        # map row indices to dataset id
        self.dataset_ccum = np.zeros(
            len(self.metadata),
        )
        # Maps dataset ids to dataset names (used in the metadata dict)
        self.dataset_map = {}
        count = 0
        for i, k in enumerate(self.metadata.keys()):
            self.dataset_ccum[i] = count
            self.dataset_map[i] = k
            count += self.metadata[k]["shape"][0]
        self.dataset_ccum[0] = -1

    def __len__(self):
        return self.num_samples

    def metadata_lookup(self, idx) -> dict:
        did = sum(~(self.dataset_ccum > idx)) - 1
        metadata = self.metadata[self.dataset_map[did]]
        return metadata

    def lookup_cell_by_idx(self, idx) -> tuple[np.array, np.array, dict]:
        ptr = slice(int(self.gene_data_ptr[idx]), int(self.gene_data_ptr[idx + 1]))
        col_idxs = np.asarray(self.gene_data_indices[ptr]).astype(int)
        gene_data = np.asarray(self.gene_data[col_idxs]).astype(np.int64)
        metadata = self.metadata_lookup(idx)
        return gene_data, col_idxs, metadata

    def __getitem__(self, idx: int) -> dict[str, Any]:
        '''Performs a lookup and the required transformation for the model'''
        gene_data, col_idxs, metadata = self.lookup_cell_by_idx(idx)
        return process_item(
            gene_data, col_idxs, metadata, self.tokenizer, gene_median=self.gene_medians, max_len=self.max_len
        )


class Item(TypedDict):
    text: np.array
    types: np.array
    padding_mask: np.array
    labels: np.array
    loss_mask: np.array
    is_random: np.array


def process_item(
    gene_data: np.array,
    gene_idxs: np.array,
    metadata: dict[str, float],
    tokenizer: Tokenizer,
    gene_median: dict = None,
    max_len: int = 1024,
    mask_prob: float = 0.15,
    target_sum: int = 10000,
    normalize: bool = True,
) -> Item:
    """Process a single item in the dataset.

    Optionally performs median normalization and rank ordering. The tokenizers CLS token is added to the beginning
    of every sample. Converts gene names to ensemble ids before tokenizing. Expects gene_medians to contain ensembl ids as keys.

    Args:
        gene_data (list): List of gene data, these are expression counts.
        gene_idxs (list): List of gene indices, these are keys in 'metadata['feature_names']' and correspdong the CSR entry. These are computed by sc_memmap.
        metadata (dict): Metadata dictionary.
        tokenizer (Tokenizer): Tokenizer object.
        gene_median (optional(dict)): Dictionary of gene medians. Defaults to None. Expects ensembl IDs to be keys.
        max_len (int): Maximum length of the item. Defaults to 1024. Applies padding to any sequence shorter than max_len and truncates any sequence longer than max_len.
        mask_prob (float): Probability of masking a token. Defaults to 0.15.
        target_sum (int): Target sum for normalization. Defaults to 10000.
        normalize (bool): Flag to normalize the gene data. Defaults to True.
            When set, this re-orders the gene tokens by their median expression value.

    Returns:
        dict: Processed item dictionary.

    NOTE: this method is very important and very useful. To generalize thiswwe should add an abstraction for
        Datasets that have some kind of functor transformation.
    """
    if max_len < 1:
        raise ValueError(f"max_len must be greater than 1, {max_len=}")

    if gene_median is None:
        gene_median = {}

    max_len = max_len - 1  # - minus 1 for [CLS] token
    # - convert from data vocab to global vocab
    gene_names = [metadata["feature_names"][idx] for idx in gene_idxs]
    genes, tokens, medians = [], [], []
    for tok, gene in zip(gene_names, gene_data):
        if tok in tokenizer.vocab:
            tokens.append(tokenizer.token_to_id(tok))
            genes.append(gene)
            if normalize:
                ens = tokenizer.gene_tok_to_ens(tok)  # Gene name to ensembl id.
                med = gene_median.get(ens, "1")  # If not in the dictionary we default to no normalization ("1")
                medians.append(med)

    genes = np.asarray(genes)
    token_ids = np.asarray(tokens)
    medians = np.asarray(medians)

    if normalize and gene_median is not None:
        # re-order according to expression median normalized rank. ascending order.
        genes = genes / genes.sum() * target_sum
        genes = genes / medians.astype(float)
        idxs = np.argsort(genes)
        genes = genes[idxs]
        token_ids = token_ids[idxs]

    # - select max_len subset, set sample to false so it doesnt permute the already rank ordered expression values.
    token_ids = sample_or_truncate_plus_pad(
        token_ids, max_len, tokenizer.token_to_id(tokenizer.pad_token), sample=False
    )

    mask = None
    # - masked tokens
    if mask_prob > 0.0:
        probs = np.full(token_ids.shape[0], mask_prob)
        probs[token_ids == tokenizer.token_to_id(tokenizer.pad_token)] = 0.0
        mask = np.random.binomial(1, probs).astype(bool)

        # - ensure [CLS] token is not masked
        mask = np.insert(mask, 0, False)

    # - add [CLS] token
    token_ids = np.insert(token_ids, 0, tokenizer.token_to_id(tokenizer.cls_token))

    labels = np.ones(len(token_ids)) * -1
    # We abuse the scenario where mask == None
    labels[mask] = token_ids[mask]

    pad_mask = token_ids == tokenizer.token_to_id(tokenizer.pad_token)

    if mask is None:
        # If prob is set to zero, we get None for our mask, which could have unintended side effects.
        mask = np.zeros(shape=token_ids.shape, dtype=bool)

    if mask is None:
        breakpoint()

    # NeMo megatron assumes this return structure.
    item = {
        "text": token_ids.astype(np.int64),
        "types": np.zeros_like(token_ids).astype(np.int64),
        "padding_mask": pad_mask.astype(np.int64),
        "labels": labels.astype(np.int64),
        "loss_mask": mask,
        "is_random": np.zeros_like(token_ids).astype(np.int64),
    }

    return item
