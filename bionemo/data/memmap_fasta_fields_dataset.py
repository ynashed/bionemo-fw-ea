# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from typing import Optional
from dataclasses import dataclass

import numpy as np

from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import TextMemMapDataset

__all__ = ["ProteinFASTAMemmapDataset"]

class FASTAFieldsMemmapDataset(TextMemMapDataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """
    def __init__(self,
                 dataset_paths,
                 workers=None,
                 tokenizer=None,
                 sort_dataset_paths=True,
                 data_sep='\n',
                 data_fields={"data": 0},
                 ):
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=ord(">"),
            header_lines=1, # skip first line since it is not an empty sequence
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
        )
        
        self._data_fields = data_fields
        self._data_sep = data_sep

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # convert text into data
        _build_data_from_text = super()._build_data_from_text
        # extract id and sequence and tokenize (if needed)
        data = {}
        text_fields = text.split(self._data_sep)
        # FIXME: should support multiple fields in a single line
        for field_name, field_idx in self._data_fields.items():
            data[field_name] = _build_data_from_text(text_fields[field_idx].split(" ")[0])
            # data[field_name] = _build_data_from_text(text_fields[field_idx])

        return data
