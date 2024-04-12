# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import List, Optional, TypedDict

import numpy as np
import scanpy
from torch.utils.data import Dataset

from bionemo.data.singlecell.utils import sample_or_truncate_plus_pad
from tokenizers import Tokenizer


class AdamsonDataset(Dataset):
    def __init__(
        self,
        preprocessed_anndata_fn: str,
        target_gep_fn: str,
        tokenizer: Tokenizer,
        median_dict: Optional[dict[str, float]] = None,
        max_len: int = 1024,
    ):
        '''Instantiates a dataset from the preprocessed artifacts, tokenizer, and median dictionary.

        Args:
            preprocessed_anndata_fn (str): File path to the preprocessed AnnData object containing the unperturbed gene expression profiles and their CRISPR targets.
            target_gep_fn (str): File path to the npz file containing the perturbed gene expression profiles.
            tokenizer: Tokenizer object used for tokenizing the gene expression profiles.
            median_dict (dict): Dictionary containing the median values for each gene. Defaults to None.
            max_len (int): Maximum length of the gene expression profiles. Defaults to 1024.

        Notes:
            The preprocessed_anndata_fn and target_gep_fn provide the necessary information for creating a training example. These
            include:
                1) A random 'unperturbed' gene expression profile (a control).
                2) Zero or more perturbations associated with that sample.
                3) The perturbed gene expression profile (the effect).

            The 'unperturbed' gene expression profiles and perturbations are located in the preprocessed_anndata_fn, under the 'data'
            and 'obs['condition']' columns, respectively. The perturbed gene expression profiles are located in the target_gep_fn,
            which is a npz file produced after preprocessing.

        '''
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.gene_medians = median_dict

        self.data = scanpy.read_h5ad(preprocessed_anndata_fn)
        self.genes = np.asarray(self.data.var.gene_name.values).squeeze()
        self.target = np.load(target_gep_fn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # See this is where it dies., assumes there is a remapping happening here.
        sample = self.data[idx]
        target = self.target[idx]
        gene_values = sample.X

        perts = sample.obs["condition"].values[0]
        return process_item(
            gene_values.squeeze(),
            self.genes,
            self.tokenizer,
            perts,
            target,
            max_len=self.max_len,
            gene_median=self.gene_medians,
        )


class Item(TypedDict):
    input_ids: np.array
    types: np.array
    padding_mask: np.array
    target: np.array


def process_item(
    gene_values: np.array,
    gene_idxs: np.array,
    tokenizer: Tokenizer,
    perts: list[str],
    target: np.array,
    gene_median: Optional[dict] = None,
    max_len: int = 1024,
    target_sum: int = 10000,
    normalize: bool = True,
) -> Item:
    """
    Process a single item for the Adamson perturb-seq dataset.

    NOTE (SKH): this is a very specific per-item processing function that is *slightly* distinct
    from the general single cell dataset. There is no parsing of metadata for gene names, this is because we are working on a raw AnnData file that
    has the metadata associated. Pertubations are also added to the mask. Going forward, I would like to see this rewritten as a general solution for
    PERTURB-seq.

    Args:
        gene_values (list): List of gene expression values.
        gene_idxs (list): List of gene indices.
        tokenizer (Tokenizer): Tokenizer object for tokenizing gene indices.
        perts (list): List of perturbations.
        target (ndarray): Target array.
        gene_median (dict): Dictionary mapping gene indices to median values. Defaults to None.
        max_len (int): Maximum length of the tokenized sequence. Defaults to 1024.
        target_sum (int): Target sum for gene expression values normalization. Defaults to 10000.
        normalize (bool): Flag indicating whether to normalize gene expression values. Defaults to True.

    Returns:
        dict: Processed item containing input_ids, types, padding_mask, and target.
    """

    max_len = max_len - 1  # - minus 1 for [CLS] token
    genes, tokens, medians = [], [], []

    # Are gene_idxs and gene_values the same length?
    for i, (tok, gene) in enumerate(zip(gene_idxs, gene_values)):
        if tok in tokenizer.vocab:
            tokens.append(tokenizer.token_to_id(tok))
            genes.append(gene)

            if normalize:
                ens = tokenizer.gene_tok_to_ens(tok)
                med = gene_median.get(ens, "1")
                medians.append(med)

    genes = np.asarray(genes)
    token_ids = np.asarray(tokens)
    medians = np.asarray(medians)

    if normalize:
        genes = genes / genes.sum() * target_sum
        genes = genes / medians.astype(float)
        idxs = np.argsort(genes)
        genes = genes[idxs]
        token_ids = token_ids[idxs]

    # - select max_len subset
    if max_len > 0:
        token_ids = sample_or_truncate_plus_pad(token_ids, max_len, tokenizer.token_to_id(tokenizer.pad_token))

    # - add [CLS] token
    token_ids = np.insert(token_ids, 0, tokenizer.token_to_id(tokenizer.cls_token))

    mask = np.zeros(len(token_ids), dtype=bool)
    for p in _parse_pert(perts):
        if p in tokenizer.vocab:
            id = tokenizer.vocab[p]
            # add pertubed genes to our mask
            mask[token_ids == id] = True

    pad_mask = token_ids == tokenizer.token_to_id(tokenizer.pad_token)
    item = {
        "input_ids": token_ids.astype(np.int64),
        "types": mask.astype(np.int64),
        "padding_mask": pad_mask.astype(np.int64),
        "target": target.astype(np.float32),
    }

    return item


# Utility functions
def _parse_pert(pert: str) -> List[str]:
    '''Pert is either a singleton (ctrl), or its a list of pertubations joined by '+'
    Returns a list of pertubations'''
    if pert == "ctrl":
        return []
    else:
        return [p for p in pert.split("+") if p != "ctrl"]
