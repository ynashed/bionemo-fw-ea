# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Callable, Dict, Sequence, TypedDict

import numpy as np
import torch
from pytriton.decorators import batch

from bionemo.model.core.infer import M
from bionemo.triton.types_constants import HIDDENS, MASK, SEQUENCES, SeqsOrBatch
from bionemo.triton.utils import encode_str_batch


__all__: Sequence[str] = (
    "DecodeRequest",
    "DecodeResponse",
    "DecodeInferFn",
    "triton_decode_infer_fn",
)


DecodeRequest = Dict[str, np.ndarray]
"""Will contain the actual embeddings alongside the input mask.

It is common for these embeddings to be under "embeddings" or "hiddens".
While the input mask will always be under "mask".
"""


class DecodeResponse(TypedDict):
    sequences: np.ndarray


DecodeInferFn = Callable[[DecodeRequest], DecodeResponse]


def triton_decode_infer_fn(model: M, k_embedding: str = HIDDENS) -> DecodeInferFn:
    """Produces a PyTriton-compatible inference function that uses a bionemo encoder-decoder model to convert batches
    of masked embeddings back into their original protein sequences strings.
    """

    @batch
    def infer_fn(**request) -> DecodeResponse:
        """Accepts a request that adheres to the :class:`DecodeRequest` schema."""
        if k_embedding not in request:
            raise ValueError(f"Expecting embeddings to be under {k_embedding}, but only found {request.keys()=}")
        if MASK not in request:
            raise ValueError(f"Expecting mask to be under {MASK}, but only found {request.keys()=}")

        embedding = torch.Tensor(request[k_embedding])
        mask = torch.Tensor(request[MASK])

        seq_batch: SeqsOrBatch = model.hiddens_to_seq(embedding, mask)

        sequences = encode_str_batch(seq_batch)
        response: DecodeResponse = {SEQUENCES: sequences}

        return response

    return infer_fn
