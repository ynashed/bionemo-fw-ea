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


# Example proteins taken from the https://github.com/facebookresearch/esm main README.
import pytest
import torch
from transformers import AutoTokenizer


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")


def test_tokenize_protein1(tokenizer):
    our_tokens = tokenizer("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")["input_ids"][1:-1]

    # fmt: off
    esm_tokens = torch.tensor(
        [20, 15, 11,  7, 10, 16,  9, 10,  4, 15,  8, 12,  7, 10, 12,  4,  9,
         10,  8, 15,  9, 14,  7,  8,  6,  5, 16,  4,  5,  9,  9,  4,  8,  7,
          8, 10, 16,  7, 12,  7, 16, 13, 12,  5, 19,  4, 10,  8,  4,  6, 19,
         17, 12,  7,  5, 11, 14, 10,  6, 19,  7,  4,  5,  6,  6])
    # fmt: on

    torch.testing.assert_close(torch.tensor(our_tokens), esm_tokens)


def test_tokenize_protein2(tokenizer):
    our_tokens = tokenizer("KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE")["input_ids"][
        1:-1
    ]

    # fmt: off
    esm_tokens = torch.tensor(
        [15,  5,  4, 11,  5, 10, 16, 16,  9,  7, 18, 13,  4, 12, 10, 13, 21,
         12,  8, 16, 11,  6, 20, 14, 14, 11, 10,  5,  9, 12,  5, 16, 10,  4,
          6, 18, 10,  8, 14, 17,  5,  5,  9,  9, 21,  4, 15,  5,  4,  5, 10,
         15,  6,  7, 12,  9, 12,  7,  8,  6,  5,  8, 10,  6, 12, 10,  4,  4,
         16,  9,  9])
    # fmt: on

    torch.testing.assert_close(torch.tensor(our_tokens), esm_tokens)


def test_tokenize_protein2_with_mask(tokenizer):
    our_tokens = tokenizer("KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE")[
        "input_ids"
    ][1:-1]

    # fmt: off
    esm_tokens = torch.tensor(
        [15,  5,  4, 11,  5, 10, 16, 16,  9,  7, 18, 13,  4, 12, 10, 13, 32,
         12,  8, 16, 11,  6, 20, 14, 14, 11, 10,  5,  9, 12,  5, 16, 10,  4,
          6, 18, 10,  8, 14, 17,  5,  5,  9,  9, 21,  4, 15,  5,  4,  5, 10,
         15,  6,  7, 12,  9, 12,  7,  8,  6,  5,  8, 10,  6, 12, 10,  4,  4,
         16,  9,  9])
    # fmt: on

    torch.testing.assert_close(torch.tensor(our_tokens), esm_tokens)


def test_tokenize_protein3(tokenizer):
    our_tokens = tokenizer("K A <mask> I S Q")["input_ids"][1:-1]
    esm_tokens = torch.tensor([15, 5, 32, 12, 8, 16])
    torch.testing.assert_close(torch.tensor(our_tokens), esm_tokens)


def test_tokenize_non_standard_tokens(tokenizer):
    our_tokens = tokenizer(" ".join(["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]))["input_ids"][1:-1]
    esm_tokens = torch.tensor([0, 1, 2, 3, 32])
    torch.testing.assert_close(torch.tensor(our_tokens), esm_tokens)


def test_tokenize_with_invalid_token(tokenizer):
    assert tokenizer("<invalid>")["input_ids"][1:-1] == [3]


def test_tokenize_with_empty_string(tokenizer):
    assert tokenizer("")["input_ids"] == [0, 2]
