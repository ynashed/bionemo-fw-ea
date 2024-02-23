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
from typing import List

import pytest
from rdkit import Chem

from bionemo.model.molecule.megamolbart import MegaMolBARTInference
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import check_model_exists, teardown_apex_megatron_cuda


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

CHECKPOINT_PATH = os.path.join(os.environ["BIONEMO_HOME"], "models/molecule/megamolbart/megamolbart.nemo")


MAX_GEN_LEN: int = 64


@pytest.fixture(scope="module")
def config_path(bionemo_home) -> str:
    path = bionemo_home / "examples" / "molecule" / "megamolbart" / "conf"
    return str(path)


@pytest.fixture(scope="module")
def inference_model(config_path) -> MegaMolBARTInference:
    cfg = load_model_config(config_name="infer", config_path=config_path)
    initialize_distributed_parallel_state()
    # load score model
    model = MegaMolBARTInference(cfg)
    model.eval()
    yield model
    teardown_apex_megatron_cuda()


@pytest.fixture(scope="module")
def _smis() -> List[str]:
    smis = [
        'c1cc2ccccc2cc1',
        'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
    ]
    return smis


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH)


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_smis_to_hiddens(inference_model: MegaMolBARTInference, _smis: List[str]):
    hidden_state, pad_masks = inference_model.seq_to_hiddens(_smis)

    assert hidden_state is not None
    # Shape should be batch, position (max of input batch here), hidden_size
    assert len(hidden_state.shape) == 3
    assert hidden_state.shape[0] == len(_smis)
    assert hidden_state.shape[1] == max([len(s) for s in _smis])
    assert hidden_state.shape[2] == inference_model.model.cfg.encoder.hidden_size
    assert pad_masks is not None


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_smis_to_embedding(inference_model: MegaMolBARTInference, _smis: List[str]):
    embedding = inference_model.seq_to_embeddings(_smis)

    assert embedding is not None
    # Shape should be batch, hidden_size (Embeddings pool out the position axis of hiddens by some means)
    assert embedding.shape[0] == len(_smis)
    assert embedding.shape[1] == inference_model.model.cfg.encoder.hidden_size
    assert len(embedding.shape) == 2


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_hidden_to_smis(inference_model: MegaMolBARTInference, _smis: List[str]):
    hidden_state, pad_masks = inference_model.seq_to_hiddens(_smis)
    inferred_smis = inference_model.hiddens_to_seq(hidden_state, pad_masks)
    log.info(f'Input SMILES and Infered: {_smis}, {inferred_smis}')

    assert len(inferred_smis) == len(_smis)

    for smi, infered_smi in zip(_smis, inferred_smis):
        log.info(f'Input and Infered:{smi},  {infered_smi}')
        input_mol = Chem.MolFromSmiles(smi)
        infer_mol = Chem.MolFromSmiles(infered_smi)
        assert input_mol is not None and infer_mol is not None

        canonical_smi = Chem.MolToSmiles(input_mol, canonical=True)
        canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
        log.info(f'Canonical Input and Infered: {canonical_smi}, {canonical_infered_smi}')

        assert canonical_smi == canonical_infered_smi


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_greedy(inference_model: MegaMolBARTInference, _smis: List[str]):
    samples = inference_model.sample(
        num_samples=3,
        sampling_method="greedy-perturbate",
        scaled_radius=1,
        seqs=_smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    nl = "\n"
    for smi_i, samples_i in zip(_smis, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join(samples_i)}\n")
    samples_flat = [item for sublist in samples for item in sublist]
    valid_molecules = []
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))
    log.info(
        f'Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))}  valids {len(valid_molecules)}'
    )

    if len(valid_molecules) < len(samples) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_topk(inference_model: MegaMolBARTInference, _smis: List[str]):
    samples = inference_model.sample(
        num_samples=3,
        sampling_method="topkp-perturbate",
        scaled_radius=0,
        topk=4,
        temperature=2,
        topp=0.0,
        seqs=_smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    nl = "\n"
    for smi_i, samples_i in zip(_smis, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join(samples_i)}\n")
    samples_flat = [item for sublist in samples for item in sublist]
    valid_molecules = []
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))
    log.info(
        f'Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))}  valids {len(valid_molecules)}'
    )

    if len(valid_molecules) < len(samples_flat) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_sample_topp(inference_model: MegaMolBARTInference, _smis: List[str]):
    samples = inference_model.sample(
        num_samples=3,
        sampling_method="topkp-perturbate",
        scaled_radius=0,
        topk=0,
        temperature=2,
        topp=0.9,
        seqs=_smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    nl = "\n"
    for smi_i, samples_i in zip(_smis, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join(samples_i)}\n")
    valid_molecules = []
    samples_flat = [item for sublist in samples for item in sublist]
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))
    log.info(
        f'Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))}  valids {len(valid_molecules)}'
    )

    if len(valid_molecules) < len(samples_flat) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
@pytest.mark.parametrize("beam_search_method", ["beam-search-perturbate", "beam-search-single-sample"])
def test_beam_search(inference_model: MegaMolBARTInference, beam_search_method: str, _smis: List[str]):
    num_samples = 3
    beam_size = 5
    samples = inference_model.sample(
        num_samples=num_samples,
        beam_size=beam_size,  # internally beam_size will be set to num_samples
        sampling_method=beam_search_method,
        beam_alpha=0,
        seqs=_smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    assert len(samples) == len(_smis)
    assert len(samples[0]) == num_samples

    nl = "\n"
    for smi_i, samples_i in zip(_smis, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join(samples_i)}\n")

    samples_flat = [item for sublist in samples for item in sublist]
    valid_molecules = []
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))

    log.info(
        f'Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))} '
        f'valids {len(valid_molecules)}'
    )

    if len(valid_molecules) < len(samples_flat) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    if beam_search_method != "beam-search-single-sample":
        # "beam-search-single-sample" is not very good, only one gaussian is sampled and top beam_size are sampled from that.
        # otherwise test that we get at least one valid molecule.
        assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_beam_search_product(inference_model: MegaMolBARTInference, _smis: List[str]):
    num_samples = 3
    beam_size = 2
    samples = inference_model.sample(
        num_samples=num_samples,
        beam_size=beam_size,
        sampling_method="beam-search-perturbate-sample",
        beam_alpha=0,
        seqs=_smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    # Samples shoudl be batch (original) x num_samples x beam_size for this sampler
    assert len(samples) == len(_smis)
    assert len(samples[0]) == num_samples
    assert len(samples[0][0]) == beam_size

    nl = "\n"
    for smi_i, ssamples_i in zip(_smis, samples):
        log.info(f"INPUT: \n{smi_i}\n")
        log.info(f"SAMPLES: \n{nl.join([nl.join(samples_i) for samples_i in ssamples_i])}\n")

    samples_flat = [item for subsublist in samples for sublist in subsublist for item in sublist]
    valid_molecules = []
    for smi in set(samples_flat):
        isvalid = False
        mol = Chem.MolFromSmiles(smi)
        if mol:
            isvalid = True
            valid_molecules.append(smi)
        log.info(f'Sample: {smi},  {isvalid}')

    log.info('Valid Molecules' + "\n".join(valid_molecules))

    log.info(
        f'Total samples = {len(samples_flat)} unique samples {len(set(samples_flat))} '
        f'valids {len(valid_molecules)}'
    )

    if len(valid_molecules) < len(samples_flat) * 0.3:
        log.warning("TOO FEW VALID SAMPLES")
    assert len(valid_molecules) != 0


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
@pytest.mark.parametrize(
    "sampling_method",
    [
        'greedy-perturbate',
        'topkp-perturbate',
        'beam-search-perturbate',
        'beam-search-perturbate-sample',
        'beam-search-single-sample',
    ],
)
def test_interpolate(inference_model: MegaMolBARTInference, sampling_method: str, _smis: List[str]):
    interpolations = inference_model.interpolate_samples(
        sample1=_smis[0],
        sample2=_smis[1],
        num_interpolations=3,
        sampling_method=sampling_method,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    assert len(interpolations) == 3
