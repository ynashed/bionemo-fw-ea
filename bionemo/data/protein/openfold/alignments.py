# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from pathlib import Path
from typing import Dict

from nemo.utils import logging


def load_alignments_super_index(
    alignments_super_index_filepath: Path,
    verbose: bool = False,
    pprefix: str = "",
) -> Dict[str, dict]:
    if verbose:
        logging.info(f"{pprefix}Loading {repr(alignments_super_index_filepath)}...")
    with open(alignments_super_index_filepath) as f:
        alignments_super_index = json.load(f)
    if verbose:
        logging.info(
            f"{pprefix}alignments_super_index ({len(alignments_super_index)})"
            f" loaded from {repr(alignments_super_index_filepath)} successfully!"
        )
    return alignments_super_index


def load_alignments(
    alignments_super_index: Dict[str, dict],
    alignments_dirpath: Path,
    key: str,
) -> dict:
    alignments_index = alignments_super_index[key]
    alignments_db_path = alignments_dirpath / alignments_index["db"]
    alignments = {}
    with open(alignments_db_path, "rb") as f:
        for file_index in alignments_index["files"]:
            filename, start, size = file_index
            f.seek(start)
            content = f.read(size).decode("utf-8")
            alignments[filename] = content
    return alignments
