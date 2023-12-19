# Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
import shutil
import time
from pathlib import Path
from subprocess import Popen

import pytest
from testbook import testbook


MODEL_NAME = "prott5nv"


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
