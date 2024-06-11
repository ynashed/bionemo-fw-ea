# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Iterable, List, Union

from nemo.collections.common.tokenizers import TokenizerSpec


__all__ = ["Label2IDTokenizer"]


class Label2IDTokenizer(TokenizerSpec):
    """Initializes simple Char Tokenizer.
    Intended to be used for extracting class labels
    for classification models such as secondary
    structure prediction model, where each class is
    encoded with a character (ex. "C", "H", "E")

        Examples:
            >>> tokenizer = Label2IDTokenizer()
            >>> seqs = ['CHE', 'CCC', 'EHH']
            >>> tokenizer = tokenizer.build_vocab(s)

    """

    def __init__(
        self,
    ):
        self.vocab = {}
        self._update_index()

    def _update_index(self):
        """
        Updates the id_to_vocab index based on the current vocab
        """
        self.decode_vocab = {id_: token for token, id_ in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocab being used."""
        return len(self.vocab)

    def text_to_tokens(self, text: str) -> List[str]:
        return list(text)

    def tokens_to_text(self, tokens):
        return "".join(tokens)

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indexes/ids
        Args:
            tokens (List[str]):  Containing tokens
        Returns:
            (List[int]): Containing ID's for each token
        """
        ids = []
        for token in tokens:
            id_ = self.vocab.get(token)
            if id_ is None:
                raise ValueError(f'Do not recognize token: {token}')
            else:
                ids.append(id_)
        return ids

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert Ids to tokens
        Args:
            ids (List[int]): Containg ids for each token
        Returns:
            (List[str]): Containing tokens
        """
        tokens = []
        for id_ in ids:
            token = self.decode_vocab.get(id_)
            if token is None:
                raise ValueError(f'Do not recognize ID: {id_}')
            tokens.append(token)
        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Converts text to ids
        Args:
            text (str): String containing text to convert
        Returns:
            (List[int]): Id's corresponding to the tokenization
            of the text
        """
        tokens = self.text_to_tokens(text)
        return self.tokens_to_ids(tokens)

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        return self.tokens_to_text(tokens)

    def build_vocab(self, strings: Union[str, Iterable[str]]):
        """Builds the vocabulary of the tokenizer from strings
        Args:
            strings: (Union[str, Iterable[str]]): Strings to
                build the vocabulary with. If a string is supplied,
                then the vocabulary is built from the single string.
                Otherwise, the vocabulary is progressively built
                from all of the strings in `strings`.
        """

        if isinstance(strings, str):
            strings = [strings]

        for string in strings:
            for token in string:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.decode_vocab[self.vocab[token]] = token

        return self
