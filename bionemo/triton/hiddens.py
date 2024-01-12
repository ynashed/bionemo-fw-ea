# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Callable, Sequence, TypedDict

import numpy as np
from pytriton.decorators import batch

from bionemo.model.core.infer import M
from bionemo.triton.types_constants import HIDDENS, MASK
from bionemo.triton.utils import decode_str_batch


__all_: Sequence[str] = (
    "HiddenResponse",
    "HiddenInferFn",
    "triton_hidden_infer_fn",
)


class HiddenResponse(TypedDict):
    hiddens: np.ndarray
    mask: np.ndarray


HiddenInferFn = Callable[[np.ndarray], HiddenResponse]


def triton_hidden_infer_fn(model: M) -> HiddenInferFn:
    @batch
    def infer_fn(sequences: np.ndarray) -> HiddenResponse:
        seqs = decode_str_batch(sequences)

        hidden, mask = model.seq_to_hiddens(seqs)

        response: HiddenResponse = {
            HIDDENS: hidden.detach().cpu().numpy(),
            MASK: mask.detach().cpu().numpy(),
        }
        return response

    return infer_fn
