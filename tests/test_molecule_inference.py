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
from pathlib import Path
from typing import List, Type

import pytest
from rdkit import Chem

from bionemo.model.molecule.infer import MolInference
from bionemo.model.molecule.megamolbart import MegaMolBARTInference
from bionemo.model.molecule.molmim.infer import MolMIMInference
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    distributed_model_parallel_state,
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@pytest.fixture(scope="module")
def _smis() -> List[str]:
    smis = [
        'c1cc2ccccc2cc1',
        'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
    ]
    return smis


MAX_GEN_LEN: int = 64


def get_inference_class(model_name: str) -> Type[MolInference]:
    return {"megamolbart": MegaMolBARTInference, "molmim": MolMIMInference}[model_name]


def get_config_dir(bionemo_home: Path, model_name: str) -> str:
    return str(bionemo_home / "examples" / "molecule" / model_name / "conf")


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
def test_smis_to_hiddens(bionemo_home: Path, _smis: List[str], model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        hidden_state, pad_masks = inferer.seq_to_hiddens(_smis)
        assert hidden_state is not None
        # Shape should be batch, position (max of input batch here), hidden_size
        assert len(hidden_state.shape) == 3
        assert hidden_state.shape[0] == len(_smis)
        assert hidden_state.shape[2] == inferer.model.cfg.encoder.hidden_size
        if inferer.model.cfg.encoder.arch == "perceiver" or isinstance(inferer, MolMIMInference):
            # Perceiver uses a fixed length for the position state. MolMIM is defined with this arch
            #  so assert it
            assert inferer.model.cfg.encoder.arch == "perceiver"
            assert hidden_state.shape[1] == inferer.model.cfg.encoder.hidden_steps
        else:
            # Most models will have one position per input token.
            # Note the following is not true in general since token length != sequence length
            #  in general. This should technically be token length, but works for these SMIS.
            assert hidden_state.shape[1] == max([len(s) for s in _smis])
        assert pad_masks is not None
        assert pad_masks.shape == hidden_state.shape[:2]


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
def test_smis_to_embedding(bionemo_home: Path, _smis: List[str], model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        embedding = inferer.seq_to_embeddings(_smis)
        assert embedding is not None
        # Shape should be batch, hidden_size (Embeddings pool out the position axis of hiddens by some means)
        assert embedding.shape[0] == len(_smis)
        assert embedding.shape[1] == inferer.model.cfg.encoder.hidden_size
        assert len(embedding.shape) == 2


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
def test_hidden_to_smis(bionemo_home: Path, _smis: List[str], model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        hidden_state, pad_masks = inferer.seq_to_hiddens(_smis)
        infered_smis = inferer.hiddens_to_seq(hidden_state, pad_masks, override_generate_num_tokens=MAX_GEN_LEN)
        log.info(f'Input SMILES and Infered: {_smis}, {infered_smis}')

        assert len(infered_smis) == len(_smis)

        for smi, infered_smi in zip(_smis, infered_smis):
            log.info(f'Input and Infered:{smi},  {infered_smi}')
            input_mol = Chem.MolFromSmiles(smi)
            infer_mol = Chem.MolFromSmiles(infered_smi)
            assert input_mol is not None

            canonical_smi = Chem.MolToSmiles(input_mol, canonical=True)
            # FIXME remove this distinction once we have a good checkpoint for MolMIM
            if isinstance(inferer, MegaMolBARTInference):
                canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
                log.info(f'Canonical Input and Infered: {canonical_smi}, {canonical_infered_smi}')
                assert canonical_smi == canonical_infered_smi
            else:
                if infer_mol is not None:
                    canonical_infered_smi = Chem.MolToSmiles(infer_mol, canonical=True)
                    if canonical_smi != canonical_infered_smi:
                        log.warning(f"Difference in SMILE {canonical_smi} vs {canonical_infered_smi}")
                else:
                    log.warning(f"Did not get a valid smile: {infered_smi}")


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
def test_sample_greedy(bionemo_home: Path, _smis: List[str], model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        samples = inferer.sample(
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


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
def test_sample_topk(bionemo_home: Path, _smis: List[str], model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        samples = inferer.sample(
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
        if isinstance(inferer, MolMIMInference):
            # FIXME remove this distinction once we have a good checkpoint
            if len(valid_molecules) == 0:
                log.warning("Found no valid molecules from MolMIM")
        else:
            assert len(valid_molecules) != 0


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
def test_sample_topp(bionemo_home: Path, _smis: List[str], model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        samples = inferer.sample(
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
        if isinstance(inferer, MolMIMInference):
            # FIXME remove this distinction once we have a good checkpoint
            if len(valid_molecules) == 0:
                log.warning("Found no valid molecules from MolMIM")
        else:
            assert len(valid_molecules) != 0


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
@pytest.mark.parametrize("beam_search_method", ["beam-search-perturbate", "beam-search-single-sample"])
def test_beam_search(bionemo_home: Path, _smis: List[str], beam_search_method: str, model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        num_samples = 3
        beam_size = 5
        samples = inferer.sample(
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
        if beam_search_method != "beam-search-single-sample" and not isinstance(inferer, MolMIMInference):
            # "beam-search-single-sample" is not very good, only one gaussian is sampled and top beam_size are sampled from that.
            # otherwise test that we get at least one valid molecule.
            assert len(valid_molecules) != 0


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
def test_beam_search_product(bionemo_home: Path, _smis: List[str], model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        num_samples = 3
        beam_size = 2
        samples = inferer.sample(
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


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_name", ["molmim", "megamolbart"])
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
def test_interpolate(bionemo_home: Path, _smis: List[str], sampling_method: str, model_name: str):
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name='infer', config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg)
        interpolations = inferer.interpolate_samples(
            sample1=_smis[0],
            sample2=_smis[1],
            num_interpolations=3,
            sampling_method=sampling_method,
            hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
        )
        assert len(interpolations) == 3
