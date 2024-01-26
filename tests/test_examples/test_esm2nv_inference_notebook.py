# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import shutil
import time
from pathlib import Path
from subprocess import Popen

import pytest
from testbook import testbook


MODEL_NAME = "esm2nv"


@pytest.fixture(scope='module')
def server(bionemo_home: Path) -> Popen:
    # Must be a seperate process, otherwise runs into known error w/ meagtron's/nemo's CUDA initialization
    # for DDP becoming incompatible with Jupyter notebook's kernel process management.
    # TODO [mgreaves] Once !553 is merged, we can re-use the test_*_triton.py's direct
    #                 creation of a `Triton` process when we load it with `interactive=True`.
    triton = Popen(
        [
            shutil.which('python'),
            "bionemo/triton/inference_wrapper.py",
            "--config-path",
            str(bionemo_home / "examples" / "protein" / MODEL_NAME / "conf"),
            "--config-name",
            "infer.yaml",
        ]
    )
    time.sleep(2)  # give process a moment to start before trying to see if it will fail
    if triton.poll() is not None:
        raise ValueError("Triton server failed to start!")
    yield triton
    triton.kill()


@pytest.fixture(scope='module')
def notebook_path(bionemo_home: Path) -> Path:
    return (bionemo_home / "examples" / "protein" / MODEL_NAME / "nbs" / "Inference.ipynb").absolute()


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
def test_example_notebook(server: Popen, notebook_path: Path):
    with testbook(str(notebook_path), execute=False, timeout=60 * 5) as tb:
        if server.poll() is not None:
            raise ValueError("Triton server failed before notebook could be executed!")
        tb.execute()  # asserts are in notebook
