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
from typing import Dict, List, Optional

from .label2id_tokenizer import Label2IDTokenizer


__all__ = ["GeneTokenizer"]


class GeneTokenizer(Label2IDTokenizer):
    """Initializes the GeneTokenizer object.

    Args:
        cls_token (str): The token used for the classification task. Defaults to "[CLS]".
        mask_token (str): The token used for masking. Defaults to "[MASK]".
        pad_token (str): The token used for padding. Defaults to "[PAD]".
        sep_token (str): The token used for separating sequences. Defaults to "[SEP]".
        ukw_token (str): The token used for unknown words. Defaults to "[UKW]".
        other_tokens (Optional[List[str]]): A list of additional special tokens. Defaults to None.
    """

    def __init__(
        self,
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        ukw_token: str = "[UKW]",
        other_tokens: Optional[List[str]] = None,
    ):
        self.vocab = {}
        self._update_index()

        self.cls_token = cls_token
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.ukw_token = ukw_token

        self.add_special_tokens([cls_token, mask_token, pad_token, sep_token, ukw_token])

        if other_tokens:
            self.add_special_tokens(other_tokens)

    def token_to_id(self, token: str) -> int:
        """
        Converts a token to its corresponding ID.

        Args:
            token (str): The token to be converted.

        Returns:
            int: The ID corresponding to the token.
        """
        return self.vocab.get(token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id(self.pad_token)

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return super().tokens_to_ids(tokens)

    def save_vocab(self, vocab_file):
        '''Saves the vocabulary as a newline delimieted vocabulary file, each line represents an int -> token mapping. line number is assumed to be the integer.'''
        vocab_dir = os.path.dirname(vocab_file)
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)

        with open(vocab_file, 'w') as f:
            for i in range(len(self.vocab)):
                f.write(self.decode_vocab[i] + '\n')

    @staticmethod
    def from_vocab_file(vocab_file):
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file {vocab_file} not found, run preprocessing to create it.")

        with open(vocab_file) as f:
            ids_to_text = {id: line.strip() for id, line in enumerate(f.readlines())}

        tokenizer = GeneTokenizer(
            cls_token=ids_to_text[0],
            mask_token=ids_to_text[1],
            pad_token=ids_to_text[2],
            sep_token=ids_to_text[3],
            ukw_token=ids_to_text[4],
        )
        vocab = {ids_to_text[id]: "NotAGeneID" for id in range(5, len(ids_to_text))}
        tokenizer.build_vocab(vocab)
        return tokenizer

    def gene_tok_to_ens(self, gene: str) -> str:
        """
        Converts a gene token to its corresponding Ensembl ID.

        Args:
            gene (str): The gene token to be converted.

        Returns:
            str: The Ensembl ID corresponding to the gene token.
        """
        return self.gene_to_ens[gene]

    def ens_tok_to_gene(self, ens: str) -> str:
        """
        Converts an Ensembl token to a gene name.

        Args:
            ens (str): The Ensembl token to be converted.

        Returns:
            str: The corresponding gene name.
        """
        return self.ens_to_gene[ens]

    def gene_to_ens(self, genes: List[str]) -> List[str]:
        """Converts a list of gene names to Ensembl IDs.

        Args:
            genes (List[str]): A list of gene names.

        Returns:
            List[str]: A list of corresponding Ensembl IDs.

        Raises:
            ValueError: If a gene name is not found in the gene_to_ens dictionary.
        """
        ens_ids = []
        for gene in genes:
            if gene in self.gene_to_ens:
                ens_ids.append(self.gene_to_ens[gene])
            else:
                raise ValueError(f"{gene} not found")
        return ens_ids

    def ens_to_gene(self, ensemble_ids: List[str]) -> List[str]:
        """Converts a list of ensemble IDs to gene names.

        Args:
            ensemble_ids (List[str]): A list of ensemble IDs.

        Returns:
            List[str]: A list of gene names corresponding to the ensemble IDs.

        Raises:
            ValueError: If an ensemble ID is not found in the mapping.
        """
        genes = []
        for ens_id in ensemble_ids:
            if ens_id in self.ens_to_gene:
                genes.append(self.ens_to_gene[ens_id])
            else:
                raise ValueError(f"{ens_id} not found")
        return genes

    def build_vocab(self, gene_ens: Dict[str, str]):
        """Builds the vocabulary of the tokenizer from gene to ens dict map

        Args:
            gene_ens: (Dict): genes and ensemble id map to
                build the vocabulary with. Only the keys are used.
        """
        self.gene_to_ens = gene_ens
        self.ens_to_gene = dict(zip(self.gene_to_ens.values(), self.gene_to_ens.keys()))

        for gene in gene_ens.keys():
            if gene not in self.vocab:
                self.vocab[gene] = len(self.vocab)
                self.decode_vocab[self.vocab[gene]] = gene

        return self

    def add_special_tokens(self, tokens: List[str]) -> None:
        """Add a series of tokens if not already present in the vocabulary

        Args:
            tokens (List[str]):
                A list of string values to be added to the vocabulary

        Returns:
            None
        """
        for token in tokens:
            if token is not None and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.decode_vocab[self.vocab[token]] = token
