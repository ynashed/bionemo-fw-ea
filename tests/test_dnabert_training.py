# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import pytest
from hydra import compose, initialize

from bionemo.model.dna.dnabert import DNABERTModel
from bionemo.model.utils import setup_trainer


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.needs_gpu
def test_dnabert_fast_dev_run():
    with initialize(config_path="./conf"):
        cfg = compose(config_name="dnabert_test")

    trainer = setup_trainer(cfg)
    model = DNABERTModel(cfg.model, trainer)
    # TODO not committing to any specific values right now, but there
    #  should ultimately be some more explicit check here
    trainer.fit(model)
