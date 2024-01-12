# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from pathlib import Path

import pytest
from testbook import testbook


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    'notebook',
    [
        "examples/protein/prott5nv/nbs/Inference_interactive.ipynb",
        "examples/protein/esm1nv/nbs/Inference_interactive.ipynb",
        "examples/protein/esm2nv/nbs/Inference_interactive.ipynb",
        "examples/molecule/megamolbart/nbs/Inference_interactive.ipynb",
    ],
)
def test_example_notebook(bionemo_home: Path, notebook: str):
    notebook_path = bionemo_home / notebook
    assert notebook_path.is_file(), f"Expected interactive notebook to exist: {notebook_path=}"
    with testbook(str(notebook_path), execute=False, timeout=60 * 5) as tb:
        tb.execute()  # asserts are in notebook
