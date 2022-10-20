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

import logging
from contextlib import contextmanager

from hydra import compose, initialize
from bionemo.model.protein.esm1nv import ESM1nvInference

log = logging.getLogger(__name__)

_INFERER = None
CONFIG_PATH = "../examples/protein/esm1nv/conf"


@contextmanager
def load_model(inf_cfg):

    global _INFERER
    if _INFERER is None:
        _INFERER = ESM1nvInference(inf_cfg)
    yield _INFERER


def test_seq_to_embedding():
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            seqs = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
                    'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF']
            embedding = inferer.seq_to_embedding(seqs)
            assert embedding is not None
            assert embedding.shape[0] == len(seqs)
            assert len(embedding.shape) == 2


def test_long_seq_to_embedding():

    long_seq = 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF'
    long_seq = long_seq * 10
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            seqs = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
                    long_seq]
            try:
                inferer.seq_to_embedding(seqs)
                assert False
            except Exception:
                pass