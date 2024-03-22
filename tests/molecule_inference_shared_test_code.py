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
from pathlib import Path
from typing import List, Type

import torch
import torch.utils
from rdkit import Chem

from bionemo.model.molecule.infer import MolInference
from bionemo.model.molecule.megamolbart import MegaMolBARTInference
from bionemo.model.molecule.molmim.infer import MolMIMInference


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

SMIS_FOR_TEST = [
    'c1cc2ccccc2cc1',
    'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC',
]


MAX_GEN_LEN: int = 64
UPDATE_GOLDEN_VALUES = os.environ.get("UPDATE_GOLDEN_VALUES", "0") == "1"


def get_inference_class(model_name: str) -> Type[MolInference]:
    return {"megamolbart": MegaMolBARTInference, "molmim": MolMIMInference}[model_name]


def get_config_dir(bionemo_home: Path, model_name: str) -> str:
    return str(bionemo_home / "examples" / "molecule" / model_name / "conf")


def get_expected_vals_file(bionemo_home: Path, model_name: str) -> Path:
    return bionemo_home / "tests" / "data" / model_name / "inference_test_golden_values.pt"


def run_smis_to_hiddens_with_goldens(inferer: MolInference, smis: List[str], expected_vals_path: Path):
    if UPDATE_GOLDEN_VALUES:
        os.makedirs(os.path.dirname(expected_vals_path), exist_ok=True)
    else:
        assert os.path.exists(
            expected_vals_path
        ), f"Expected values file not found at {expected_vals_path}. Rerun with UPDATE_GOLDEN_VALUES=1 to create it."
    assert inferer.training is False
    hidden_state, pad_masks = inferer.seq_to_hiddens(smis)
    hidden_state2, pad_masks2 = inferer.seq_to_hiddens(smis)
    hidden_state3, pad_masks3 = inferer.seq_to_hiddens(smis)
    assert hidden_state is not None
    assert hidden_state2 is not None
    assert hidden_state3 is not None
    # Shape should be batch, position (max of input batch here), hidden_size
    assert len(hidden_state.shape) == 3
    assert hidden_state.shape[0] == len(smis)
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
        assert hidden_state.shape[1] == max([len(s) for s in smis])
    assert pad_masks is not None
    assert pad_masks2 is not None
    assert pad_masks3 is not None
    assert pad_masks.shape == hidden_state.shape[:2]

    # Make sure that sequential runs of infer give the same result.
    torch.testing.assert_close(pad_masks3, pad_masks2, rtol=None, atol=None, equal_nan=True)
    torch.testing.assert_close(pad_masks, pad_masks2, rtol=None, atol=None, equal_nan=True)
    torch.testing.assert_close(hidden_state3, hidden_state2, rtol=None, atol=None, equal_nan=True)
    torch.testing.assert_close(hidden_state, hidden_state2, rtol=None, atol=None, equal_nan=True)

    if UPDATE_GOLDEN_VALUES:
        torch.save(
            {
                "expected_hidden_state": hidden_state,
                "expected_pad_masks": pad_masks,
            },
            expected_vals_path,
        )
        assert False, f"Updated expected values at {expected_vals_path}, rerun with UPDATE_GOLDEN_VALUES=0"
    else:
        expected_vals = {k: v.to(pad_masks.device) for k, v in torch.load(expected_vals_path).items()}
        torch.testing.assert_close(
            hidden_state, expected_vals["expected_hidden_state"], rtol=None, atol=None, equal_nan=True
        )
        assert torch.all(pad_masks == expected_vals["expected_pad_masks"])


def run_smis_to_embedding(inferer: MolInference, smis: List[str]):
    embedding = inferer.seq_to_embeddings(smis)
    assert embedding is not None
    # Shape should be batch, hidden_size (Embeddings pool out the position axis of hiddens by some means)
    assert embedding.shape[0] == len(smis)
    assert embedding.shape[1] == inferer.model.cfg.encoder.hidden_size
    assert len(embedding.shape) == 2


def run_hidden_to_smis(inferer: MolInference, smis: List[str]):
    hidden_state, pad_masks = inferer.seq_to_hiddens(smis)
    infered_smis = inferer.hiddens_to_seq(hidden_state, pad_masks, override_generate_num_tokens=MAX_GEN_LEN)
    log.info(f'Input SMILES and Infered: {smis}, {infered_smis}')

    assert len(infered_smis) == len(smis)

    for smi, infered_smi in zip(smis, infered_smis):
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


def run_sample_not_beam(inferer: MolInference, smis: List[str], sampling_method: str):
    samples = inferer.sample(
        num_samples=3,
        sampling_method=sampling_method,
        scaled_radius=0,
        topk=4,
        temperature=2,
        topp=0.0,
        seqs=smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    nl = "\n"
    for smi_i, samples_i in zip(smis, samples):
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


def run_beam_search(inferer: MolInference, smis: List[str], beam_search_method: str):
    num_samples = 3
    beam_size = 5
    samples = inferer.sample(
        num_samples=num_samples,
        beam_size=beam_size,  # internally beam_size will be set to num_samples
        sampling_method=beam_search_method,
        beam_alpha=0,
        seqs=smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    assert len(samples) == len(smis)
    assert len(samples[0]) == num_samples

    nl = "\n"
    for smi_i, samples_i in zip(smis, samples):
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


def run_beam_search_product(inferer: MolInference, smis: List[str]):
    num_samples = 3
    beam_size = 2
    samples = inferer.sample(
        num_samples=num_samples,
        beam_size=beam_size,
        sampling_method="beam-search-perturbate-sample",
        beam_alpha=0,
        seqs=smis,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    # Samples shoudl be batch (original) x num_samples x beam_size for this sampler
    assert len(samples) == len(smis)
    assert len(samples[0]) == num_samples
    assert len(samples[0][0]) == beam_size

    nl = "\n"
    for smi_i, ssamples_i in zip(smis, samples):
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


def run_interpolate(inferer: MolInference, smis: List[str], sampling_method: str):
    interpolations = inferer.interpolate_samples(
        sample1=smis[0],
        sample2=smis[1],
        num_interpolations=3,
        sampling_method=sampling_method,
        hiddens_to_seq_kwargs={"override_generate_num_tokens": MAX_GEN_LEN},
    )
    assert len(interpolations) == 3
