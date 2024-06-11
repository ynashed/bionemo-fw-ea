# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np


def sample_or_truncate_plus_pad(
    gene_ids: np.array,
    max_length: int,
    pad_token_id: int,
    sample: bool = True,
) -> np.array:
    """Truncate and pad samples.

    Args:
        gene_ids (np.ndarray): Array of gene IDs.
        max_length (int): Maximum length of the samples.
        pad_token_id (int): ID of the padding token.
        sample (bool, optional): Whether to sample or truncate the samples. Defaults to True.

    Returns:
        np.array: Tuple containing the truncated or padded gene IDs.
    """
    if len(gene_ids) == max_length:
        return gene_ids

    if len(gene_ids) > max_length:  # - sample or truncate
        if sample:
            indices = np.random.permutation(len(gene_ids))[:max_length]
            return gene_ids[indices]
        else:
            return gene_ids[:max_length]
    else:  # - pad
        pad_tokens = np.full((max_length - len(gene_ids)), pad_token_id, dtype=np.int32)
        gene_ids = np.concatenate([gene_ids, pad_tokens])
        return gene_ids
