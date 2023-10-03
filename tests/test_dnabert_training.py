# Copyright (c) 2022, NVIDIA CORPORATION.
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

import os

import pytest
from hydra import compose, initialize

from bionemo.model.dna.dnabert import DNABERTModel
from bionemo.model.utils import setup_trainer


@pytest.mark.slow
@pytest.mark.needs_gpu
def test_dnabert_fast_dev_run():
    os.environ["PROJECT_MOUNT"] = os.environ.get("PROJECT_MOUNT", '/workspace/bionemo')

    with initialize(config_path="./conf"):
        cfg = compose(config_name="dnabert_test")

    trainer = setup_trainer(cfg)
    model = DNABERTModel(cfg.model, trainer)
    # TODO not committing to any specific values right now, but there
    #  should ultimately be some more explicit check here
    trainer.fit(model)
