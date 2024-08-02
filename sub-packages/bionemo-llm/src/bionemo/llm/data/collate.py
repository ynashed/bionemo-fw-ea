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


from typing import Sequence

import torch

from bionemo.llm.data import types


def collate_fn(
    batch: Sequence[types.BertSample],
    padding_value: int,
    max_length: int | None = None,
) -> types.BertSample:
    """Collate function with padding.

    Args:
        batch (list): List of samples.
        padding_value (int, optional): Padding value. Defaults to 0.
        max_length (int, optional): Maximum length of the sequence. If not provided, the maximum length of the batch
            will be used. Tensors longer than this value will be truncated. Defaults to None.

    Returns:
        A collated batch with the same dictionary input structure.
    """

    def _pad(tensors, padding_value):
        if max_length is not None:
            tensors = [t[:max_length] for t in tensors]
        batched_tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        if max_length is None:
            return batched_tensors
        return torch.nn.functional.pad(batched_tensors, (0, max_length - batched_tensors.size(1)), value=padding_value)

    return {
        "text": _pad([s["text"] for s in batch], padding_value),
        "types": _pad([s["types"] for s in batch], 0),
        "attention_mask": _pad([s["attention_mask"] for s in batch], False),
        "labels": _pad([s["labels"] for s in batch], -1),
        "loss_mask": _pad([s["loss_mask"] for s in batch], False),
        "is_random": _pad([s["is_random"] for s in batch], 0),
    }
