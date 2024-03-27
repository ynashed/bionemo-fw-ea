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

import pytest

from bionemo.model.protein.esm1nv import ESM1nvInference
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import check_model_exists, teardown_apex_megatron_cuda


log = logging.getLogger(__name__)
CHECKPOINT_PATH = os.path.join(os.environ["BIONEMO_HOME"], "models/protein/esm1nv/esm1nv.nemo")


@pytest.fixture(scope="module")
def config_path(bionemo_home) -> str:
    path = bionemo_home / "examples" / "protein" / "esm1nv" / "conf"
    return str(path.absolute())


@pytest.fixture(scope="module")
def inference_model(config_path) -> ESM1nvInference:
    cfg = load_model_config(config_name="infer", config_path=config_path)
    # load score model
    initialize_distributed_parallel_state()
    model = ESM1nvInference(cfg)
    model.eval()
    yield model
    teardown_apex_megatron_cuda()


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH)


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_seq_to_embedding(inference_model: ESM1nvInference):
    seqs = [
        'MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
        'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF',
    ]
    embedding, mask = inference_model.seq_to_hiddens(seqs)
    assert embedding is not None
    assert embedding.shape[0] == len(seqs)
    assert len(embedding.shape) == 3


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_long_seq_to_embedding(inference_model: ESM1nvInference):
    long_seq = 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF'
    long_seq = long_seq * 10
    seqs = [
        'MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
        long_seq,
    ]
    try:
        inference_model.seq_to_hiddens(seqs)
        assert False
    except Exception:
        pass
