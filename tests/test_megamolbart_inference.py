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
from rdkit import Chem

from hydra import compose, initialize
from hydra.core.plugins import Plugins
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
import os

from bionemo.model.molecule.megamolbart import MegaMolBARTInference

log = logging.getLogger(__name__)


_INFERER = None
CONFIG_PATH = "../examples/molecule/megamolbart/conf"
PREPEND_CONFIG_DIR = os.path.abspath("../examples/conf")

@contextmanager
def load_model(inf_cfg):

    global _INFERER
    if _INFERER is None:
        _INFERER = MegaMolBARTInference(inf_cfg)
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

def test_smis_to_hiddens():
    register_searchpath_prepend_plugin()
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            smis = ['c1cc2ccccc2cc1',
                    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
                    'CC(=O)C(=O)N1CCC([C@H]2CCCCN2C(=O)c2ccc3c(n2)CCN(C(=O)OC(C)(C)C)C3)CC1']
            hidden_state, pad_masks = inferer.seq_to_hiddens(smis)

            assert hidden_state is not None
            assert hidden_state.shape[0] == len(smis)
            assert hidden_state.shape[2] == inferer.model.cfg.max_position_embeddings
            assert pad_masks is not None


def test_smis_to_embedding():
    register_searchpath_prepend_plugin()
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            smis = ['c1cc2ccccc2cc1',
                    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
                    'CC(=O)C(=O)N1CCC([C@H]2CCCCN2C(=O)c2ccc3c(n2)CCN(C(=O)OC(C)(C)C)C3)CC1']
            embedding = inferer.seq_to_embeddings(smis)

            assert embedding is not None
            assert embedding.shape[0] == len(smis)
            assert embedding.shape[1] == inferer.model.cfg.max_position_embeddings


def test_hidden_to_smis():
    register_searchpath_prepend_plugin()
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            smis = ['c1cc2ccccc2cc1',
                    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
                    'CC(=O)C(=O)N1CCC([C@H]2CCCCN2C(=O)c2ccc3c(n2)CCN(C(=O)OC(C)(C)C)C3)CC1']
            hidden_state, pad_masks = inferer.seq_to_hiddens(smis)
            infered_smis = inferer.hiddens_to_seq(hidden_state, pad_masks)
            log.info(f'Input SMILES and Infered: {smis}, {infered_smis}')

            assert(len(infered_smis) == len(smis))

            for smi, infered_smi in zip(smis, infered_smis):
                log.info(f'Input and Infered:{smi},  {infered_smi}')
                input_mol = Chem.MolFromSmiles(smi)
                infer_mol = Chem.MolFromSmiles(infered_smi)
                assert input_mol is not None and infer_mol is not None

                canonical_smi = Chem.MolToSmiles(input_mol, canonical=True)
                canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
                log.info(f'Canonical Input and Infered: {canonical_smi}, {canonical_infered_smi}')

                assert(canonical_smi == canonical_infered_smi)


def test_sample():
    register_searchpath_prepend_plugin()
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="infer")

        with load_model(cfg) as inferer:
            smis = ['c1cc2ccccc2cc1',
                    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
                    'CC(=O)C(=O)N1CCC([C@H]2CCCCN2C(=O)c2ccc3c(n2)CCN(C(=O)OC(C)(C)C)C3)CC1']
            samples = inferer.sample(
                num_samples=3, 
                sampling_method="greedy-perturbate",
                scaled_radius=1,
                topk=4,
                smis=smis, 
            )
            samples = set(samples)
            log.info('\n'.join(smis))
            log.info('\n'.join(samples))
            valid_molecules = []
            for smi in set(samples):
                isvalid = False
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    isvalid = True
                    valid_molecules.append(smi)
                log.info(f'Sample: {smi},  {isvalid}')

            log.info('Valid Molecules' + "\n".join(valid_molecules))
            log.info(f'Total samples = {len(samples)} unique samples {len(set(samples))}  valids {len(valid_molecules)}')

            if len(valid_molecules) < len(samples) * 0.3:
                log.warning("TOO FEW VALID SAMPLES")
            assert len(valid_molecules) != 0
