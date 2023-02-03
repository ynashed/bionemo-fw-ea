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
import os

from hydra import compose, initialize
from hydra.core.plugins import Plugins
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from bionemo.model.protein.esm1nv import ESM1nvInference

log = logging.getLogger(__name__)

_INFERER = None
CONFIG_PATH = "../examples/protein/esm1nv/conf"
PREPEND_CONFIG_DIR = os.path.abspath("../examples/conf")

@contextmanager
def load_model(inf_cfg):
    global _INFERER
    if _INFERER is None:
        _INFERER = ESM1nvInference(inf_cfg)
    yield _INFERER


# TODO Move to module for use elsewhere -- requires different solution for passing prepended directory
class SearchPathPrepend(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Add search_path to the end of the existing search path
        search_path.prepend( 
            provider="searchpath-plugin", path=f"file://{PREPEND_CONFIG_DIR}"
        )


def register_searchpath_prepend_plugin() -> None:
    """Call this function before invoking @hydra.main"""
    Plugins.instance().register(SearchPathPrepend)


def test_seq_to_embedding():
    register_searchpath_prepend_plugin()
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="infer")
        with load_model(cfg) as inferer:
            seqs = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
                    'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF']
            embedding, mask = inferer.seq_to_hiddens(seqs)
            assert embedding is not None
            assert embedding.shape[0] == len(seqs)
            assert len(embedding.shape) == 3


def test_long_seq_to_embedding():
    register_searchpath_prepend_plugin()
    long_seq = 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF'
    long_seq = long_seq * 10
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            seqs = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
                    long_seq]
            try:
                inferer.seq_to_hiddens(seqs)
                assert False
            except Exception:
                pass
