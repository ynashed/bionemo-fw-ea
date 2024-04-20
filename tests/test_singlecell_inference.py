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
from copy import deepcopy
from pathlib import Path
from typing import Generator, List

import pytest

from bionemo.model.singlecell.geneformer.infer import GeneformerInference
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    distributed_model_parallel_state,
)

from .inference_shared_test_code import (
    get_config_dir,
    get_expected_vals_file,
    run_seqs_to_embedding,
    run_seqs_to_hiddens_with_goldens,
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


CELLS_FOR_TEST = [
    [
        'ACOT7',
        'ESPN',
        'CACNA1E',
        'ZNF648',
        'LINC01344',
        'YPEL5P1',
        'RNU6-152P',
        'EIF1P3',
        'MIR4794',
        'RAVER2',
        'SPATA1',
    ],
    ['SCYGR3', 'NBPF3', 'SCYGR6', 'RNA5SP114', 'MIR561', 'GULP1', 'RNU6-298P', 'TEFM', 'PFN1P10'],
    ['RNU2-38P', 'RNVU1-24', 'SLC25A34', 'SLC25A34-AS1', 'TMEM82', 'FBLIM1', 'SUZ12P1', 'CRLF3'],
]


@pytest.fixture()
def cells() -> List[List[str]]:
    return deepcopy(CELLS_FOR_TEST)


@pytest.fixture(scope="module")
def geneformer_inferer(bionemo_home: Path) -> Generator[GeneformerInference, None, None]:
    model_name = "geneformer"
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = GeneformerInference(cfg=cfg, inference_batch_size_for_warmup=2)  # Change to 1 to debug the failure
        yield inferer  # Yield so cleanup happens after the test


@pytest.fixture(scope="module")
def geneformer_expected_vals_path(bionemo_home: Path) -> Path:
    return get_expected_vals_file(bionemo_home, "geneformer")


def test_geneformer_inference_tokenization(geneformer_inferer: GeneformerInference, cells: List[List[str]]):
    cell_tokenization, genes_mask = geneformer_inferer.tokenize(cells)
    assert cell_tokenization.shape == (
        3,
        max(len(cell) for cell in cells) + 1,
    ), "Each gene should be 1 token plus one token for [CLS]"
    cls_token_id = geneformer_inferer.tokenizer.token_to_id(geneformer_inferer.tokenizer.cls_token)
    n_tokens = genes_mask.sum(-1)
    assert all([n_tokens[i] == len(cells[i]) + 1] for i in range(len(cells)))
    assert all(cell_tokenization[:, 0] == cls_token_id), "First column should be all [CLS] tokens."


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_cells_to_hiddens_with_goldens_geneformer(
    geneformer_inferer: GeneformerInference, cells: List[List[str]], geneformer_expected_vals_path: Path
):
    run_seqs_to_hiddens_with_goldens(
        geneformer_inferer,
        cells,
        geneformer_expected_vals_path,
        geneformer_inferer.model.cfg.hidden_size,
        "bert",
        geneformer_inferer._tokenize,
    )


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_cells_to_embedding_geneformer(geneformer_inferer: GeneformerInference, cells: List[List[str]]):
    run_seqs_to_embedding(geneformer_inferer, cells, geneformer_inferer.model.cfg.hidden_size)
