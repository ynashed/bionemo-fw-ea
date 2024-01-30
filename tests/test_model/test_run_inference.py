# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import pytest

from bionemo.model import run_inference
from bionemo.triton.utils import load_model_config
from bionemo.utils.tests import reset_microbatch_calculator


_TEST_DATA: List[Tuple[str, int]] = [
    ('prott5nv_infer.yaml', 768),
    ('esm1nv_infer.yaml', 768),
    ('megamolbart_infer.yaml', 512),
]
"""(config filename, hidden embedding dimensionality)"""


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.parametrize('config_name, hidden_size', _TEST_DATA)
def test_model_inference(bionemo_home: Path, config_name: str, hidden_size: int):
    """Verify that inference with each model produces the correct size for hidden weights."""

    config_path = bionemo_home / "examples" / "tests" / "conf"

    cfg = load_model_config(config_path, config_name)

    with NamedTemporaryFile() as tempfi:
        run_inference.main(config_path, config_name, output_override=tempfi.name, overwrite=True)

        with open(tempfi.name, 'rb') as rb:
            embeddings = pickle.load(rb)

    dict_keys = cfg.model.downstream_task.outputs + ["sequence", "id"]
    for emb in embeddings:
        for key in dict_keys:
            if key not in emb.keys():
                assert False, f'Missing key {key} in embeddings file {cfg.model.data.output_fname}'
        if "hiddens" in dict_keys:
            assert emb["hiddens"].shape == (len(emb["sequence"]), hidden_size)
        if "embeddings" in dict_keys:
            assert emb["embeddings"].shape == (hidden_size,)
    reset_microbatch_calculator()
