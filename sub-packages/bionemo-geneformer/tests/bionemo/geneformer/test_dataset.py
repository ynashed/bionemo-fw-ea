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

import numpy as np
import pytest

from bionemo.geneformer.data.singlecell.dataset import apply_masking


def test_masking_gives_expected_ratios():
    token_ids = np.ones(100_000, dtype=np.int64)

    masked_token_ids, loss_mask, _, _ = apply_masking(
        token_ids,
        mask_prob=0.5,
        mask_token_prob=0.25,
        random_token_prob=0.12,
        prepend_cls_token=True,
        pad_id=0,
        cls_id=5,
        mask_id=2,
        num_tokens=2,
    )

    assert len(masked_token_ids) == 100_001
    masked_token_ids = masked_token_ids[1:]
    loss_mask = loss_mask[1:]

    # Check that overall masking probability is correct.
    assert pytest.approx(loss_mask.mean(), abs=0.01) == 0.5

    # Check that the distribution of masked tokens is correct.
    assert pytest.approx((masked_token_ids == 2).mean(), abs=0.01) == 0.5 * 0.25

    # Check that the distribution of random tokens is correct.
    assert pytest.approx(((masked_token_ids == 5) | (masked_token_ids == 6)).mean(), abs=0.01) == 0.5 * 0.12

    # Check that the distribution of unmasked tokens is correct.
    assert pytest.approx((masked_token_ids[loss_mask] == 1).mean(), abs=0.01) == 1.0 - (0.25 + 0.12)


def test_binomial_logic():
    token_ids = np.ones(100_000, dtype=np.int64)
    mask_prob = 1.0
    mask_token_prob = 0.50
    random_token_prob = 0.50
    pad_id = 0

    probs = np.full(token_ids.shape[0], mask_prob)
    probs[token_ids == pad_id] = 0.0
    mask = np.random.binomial(1, probs).astype(bool)
    mask_tokens_positions = np.random.binomial(1, mask_token_prob, mask.shape).astype(bool)

    random_tokens_positions = ~mask_tokens_positions & np.random.binomial(1, random_token_prob, mask.shape).astype(
        bool
    )

    assert pytest.approx(mask.mean(), abs=0.01) == mask_prob
    assert pytest.approx(mask_tokens_positions.mean(), abs=0.01) == mask_prob * mask_token_prob
    assert pytest.approx(random_tokens_positions.mean(), abs=0.01) == mask_prob * random_token_prob
