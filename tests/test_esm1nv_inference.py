# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import pathlib
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from bionemo.model.protein.esm1nv import ESM1nvInference
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    check_model_exists,
    register_searchpath_config_plugin,
    reset_microbatch_calculator,
    update_relative_config_dir,
)


log = logging.getLogger(__name__)

CHECKPOINT_PATH = os.path.join(os.getenv("BIONEMO_HOME"), "models/protein/esm1nv/esm1nv.nemo")


@pytest.fixture(scope='module')
def infer_cfg() -> DictConfig:
    # TODO(dorotat): Figure out how to import this method from with correctly setup paths, especially this_file_dir
    config_path = "examples/protein/esm1nv/conf"
    config_name = "infer"
    prepend_config_dir = os.path.join(os.getenv("BIONEMO_HOME"), "examples/conf")
    this_file_dir = pathlib.Path(pathlib.Path(os.path.abspath(__file__)).parent)
    absolute_config_path = os.path.join(os.getenv("BIONEMO_HOME"), config_path)
    relative_config_path = os.path.relpath(absolute_config_path, this_file_dir)

    # TODO(dorotat): figure out more elegant way which can be be externalise to add search path to hydra
    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(Path(prepend_config_dir), this_file_dir)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=relative_config_path):
        cfg = compose(config_name=config_name)
    yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture(scope='module')
def inferer(infer_cfg: DictConfig) -> ESM1nvInference:
    yield ESM1nvInference(infer_cfg)
    reset_microbatch_calculator()


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH)


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_seq_to_embedding(inferer: ESM1nvInference):
    seqs = [
        'MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
        'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF',
    ]
    embedding, mask = inferer.seq_to_hiddens(seqs)
    assert embedding is not None
    assert embedding.shape[0] == len(seqs)
    assert len(embedding.shape) == 3


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_long_seq_to_embedding(inferer: ESM1nvInference):
    long_seq = 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF'
    long_seq = long_seq * 10
    seqs = [
        'MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
        long_seq,
    ]
    try:
        inferer.seq_to_hiddens(seqs)
        assert False
    except Exception:
        pass
