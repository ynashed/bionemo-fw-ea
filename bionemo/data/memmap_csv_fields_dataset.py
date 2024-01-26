# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re

from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import TextMemMapDataset
from nemo.utils import logging


__all__ = ["CSVFieldsMemmapDataset"]


class CSVFieldsMemmapDataset(TextMemMapDataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    Returns a dictionary with multiple fields.

    WARNING: This class has been migrated to NeMo and will be removed from BioNeMo when NeMo container 1.21 is used.
    Every change to this class should be added to the class in NeMo.
    https://github.com/NVIDIA/NeMo/blob/83d6614fbf29cf885f3bc36233f6e3758ba8f1e3/nemo/collections/nlp/data/language_modeling/text_memmap_dataset.py#L336

    """

    def __init__(
        self,
        dataset_paths,
        newline_int=10,
        header_lines=1,
        workers=None,
        tokenizer=None,
        sort_dataset_paths=True,
        # data_fields - dict of field names and their corresponding indices
        data_sep=',',
        data_fields={"data": 0},
        index_mapping_dir=None,
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=newline_int,
            header_lines=header_lines,
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )

        self._data_fields = data_fields
        self._data_sep = data_sep
        logging.warning("CSVFieldsMemmapDataset will be available in NeMo 1.21")

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # convert text into data
        _build_data_from_text = super()._build_data_from_text
        # extract id and sequence and tokenize (if needed)
        data = {}
        rule = self._data_sep + r'\s*(?=(?:[^"]*"[^"]*")*[^"]*$)'
        text_fields = re.split(r'{}'.format(rule), text)
        for field_name, field_idx in self._data_fields.items():
            data[field_name] = _build_data_from_text(text_fields[field_idx].strip('\"').strip())

        return data
