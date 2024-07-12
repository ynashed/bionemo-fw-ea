# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Sequence

from fw2nim.config.types import BaseModel
from pydantic import Field


__all__: Sequence[str] = (
    "MolMimSequences",
    "MolMimEmbeddings",
    "MolMimGenerated",
    "MolMimHiddens",
    "MolMimControlOptIn",
    "MolMimControlOptOut",
)


class MolMimSequences(BaseModel):
    sequences: list[str] = Field(..., numpy_dtype="bytes", triton_shape=(-1,))


class MolMimEmbeddings(BaseModel):
    embeddings: list[list[float]] = Field(
        ...,
        numpy_dtype="float32",
        triton_shape=(-1, 512),
    )


class MolMimGenerated(BaseModel):
    generated: list[str] = Field(
        ...,
        numpy_dtype="bytes",
        triton_shape=(-1,),
    )


class MolMimHiddens(BaseModel):
    hiddens: list[list[list[float]]] = Field(
        ...,
        numpy_dtype="float32",
        triton_shape=(-1, 512),
    )
    mask: list[list[bool]] = Field(
        ...,
        numpy_dtype="bool",
        triton_shape=(-1,),
    )


class MolMimControlOptIn(BaseModel):
    smi: str = Field(
        ...,
        numpy_dtype="bytes",
        rows=True,
    )
    algorithm: str = Field(..., numpy_dtype="bytes", triton_shape=(1, 1))
    num_molecules: int = Field(..., numpy_dtype="int32", triton_shape=(1, 1))
    property_name: str = Field(..., numpy_dtype="bytes", triton_shape=(1, 1))
    minimize: bool = Field(..., numpy_dtype="bool", triton_shape=(1, 1))
    min_similarity: float = Field(..., numpy_dtype="float32", triton_shape=(1, 1))
    particles: int = Field(..., numpy_dtype="int32", triton_shape=(1, 1))
    iterations: int = Field(..., numpy_dtype="int32", triton_shape=(1, 1))
    radius: float = Field(..., numpy_dtype="float32", triton_shape=(1, 1))


class MolMimControlOptOut(BaseModel):
    samples: list[str] = Field(
        ...,
        numpy_dtype="bytes",
        rows=True,
    )
    scores: list[float] = Field(
        ...,
        numpy_dtype="float32",
        triton_shape=(-1,),
        rows=True,
    )
    score_type: str = Field(..., numpy_dtype="bytes", triton_shape=(1, 1))
