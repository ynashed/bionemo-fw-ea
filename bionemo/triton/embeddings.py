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
import torch
from model_navigator.package.package import Package
from pytriton.decorators import batch

from bionemo.model.core.infer import M
from bionemo.model.protein.esm1nv.esm1nv_model import ESM1nvModel
from bionemo.triton.types_constants import EMBEDDINGS, StrArray
from bionemo.triton.utils import decode_str_batch


__all_: Sequence[str] = (
    "NavEmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingInferFn",
    "triton_embedding_infer_fn",
    "nav_triton_embedding_infer_fn",
    "mask_postprocessing_fn",
)


class NavEmbeddingRequest(TypedDict):
    tokens: np.ndarray
    mask: np.ndarray


class EmbeddingResponse(TypedDict):
    embeddings: np.ndarray


EmbeddingInferFn = Callable[[StrArray], EmbeddingResponse]


def triton_embedding_infer_fn(model: M) -> EmbeddingInferFn:
    @batch
    def infer_fn(sequences: np.ndarray) -> EmbeddingResponse:
        seqs = decode_str_batch(sequences)

        embedding = model.seq_to_embeddings(seqs)

        response: EmbeddingResponse = {
            EMBEDDINGS: embedding.detach().cpu().numpy(),
        }
        return response

    return infer_fn


def nav_triton_embedding_infer_fn(model: M, runner: Package) -> EmbeddingInferFn:
    postprocess = mask_postprocessing_fn(model)

    @batch
    def infer_fn(sequences: np.ndarray) -> EmbeddingResponse:
        seqs = decode_str_batch(sequences)

        tokens_enc, enc_mask = model.tokenize(seqs)
        inp: NavEmbeddingRequest = {
            'tokens': tokens_enc.cpu().detach().numpy(),
            'mask': enc_mask.cpu().detach().numpy(),
        }

        hidden_states = runner.infer(inp)
        hidden_states = torch.tensor(hidden_states['embeddings'], device='cuda')
        enc_mask = postprocess(enc_mask)

        embedding = model.hiddens_to_embedding(hidden_states, enc_mask)

        response: EmbeddingResponse = {
            EMBEDDINGS: embedding.cpu().numpy(),
        }
        return response

    return infer_fn


def mask_postprocessing_fn(model: M) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(model, ESM1nvModel):

        def postprocess(enc_mask: torch.Tensor) -> torch.Tensor:
            enc_mask[:, 0:2] = 0
            enc_mask = torch.roll(enc_mask, shifts=-1, dims=1)
            return enc_mask

    else:

        def postprocess(enc_mask: torch.Tensor) -> torch.Tensor:
            return enc_mask

    return postprocess
