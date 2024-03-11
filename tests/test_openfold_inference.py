import os
import tempfile
from pathlib import Path
from typing import Iterator, Tuple

import pytest
import pytorch_lightning as plt
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bionemo.data.protein.openfold.datahub import get_structured_paths
from bionemo.data.protein.openfold.datasets import PredictDataset
from bionemo.data.protein.openfold.features import create_mmcif_features
from bionemo.data.protein.openfold.helpers import collate
from bionemo.data.protein.openfold.mmcif import load_mmcif_file, parse_mmcif_string
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.validation_metrics import compute_validation_metrics
from bionemo.model.protein.openfold.writer import PredictionFeatureWriter
from bionemo.model.utils import setup_trainer


GRADIENT_CHECKPOINTING = False

INFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../examples/tests/test_data/openfold_data/inference'
)

PDB_DIR = os.path.join(INFERENCE_DIR, 'pdb')
PDB_NAMES = ['first_sequence.pdb', 'second_sequence.pdb']
PDB_PATHS = [Path(os.path.join(PDB_DIR, pdb)) for pdb in PDB_NAMES]

CIF_NAMES = ['first_sequence.cif', 'second_sequence.cif']
CIF_PATHS = [Path(os.path.join(PDB_DIR, cif)) for cif in CIF_NAMES]
CIF_CHAIN_IDS = ["A", "B"]

MSA_DIR = os.path.join(INFERENCE_DIR, 'msas')
MSA_NAMES = [
    ['7ZHL_A_mgnify_alignment.a3m', '7ZHL_A_smallbfd_alignment.a3m', '7ZHL_A_uniref90_alignment.a3m'],
    ['7YVT_B_mgnify_alignment.a3m', '7YVT_B_smallbfd_alignment.a3m', '7YVT_B_uniref90_alignment.a3m'],
]
MSA_PATHS = [[os.path.join(MSA_DIR, msa) for msa in msas] for msas in MSA_NAMES]

PT_DIR = os.path.join(INFERENCE_DIR, 'pt')
PT_NAMES = ['first_sequence.pt', 'second_sequence.pt']
PT_PATHS = [Path(os.path.join(PT_DIR, pt)) for pt in PT_NAMES]


@pytest.fixture(scope='module')
def infer_cfg() -> Iterator[DictConfig]:
    """Setting up the general inference config object..

    Yields:
        Iterator[DictConfig]: Inference Config object containing path and name
    """
    this_file_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = "examples/protein/openfold/conf"
    config_name = "infer"
    absolute_config_path = os.path.join(os.getenv("BIONEMO_HOME"), config_path)
    relative_config_path = os.path.relpath(absolute_config_path, this_file_dir)
    with initialize(config_path=relative_config_path):
        cfg = compose(config_name=config_name)
    yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture(scope='function')
def alphafold_cfg(request, infer_cfg) -> DictConfig:
    """Append additional parameters to the inference config specific to the instantiated AF model.

    Args:
        request: Request object containing the parameters that should be updated.
        infer_cfg (DictConfig): Initial inference config object.

    Returns:
        DictConfig: Updated AlphaFold config.
    """
    msa_a3m_filepaths, generate_templates_if_missing = request.param
    infer_cfg.model.data.msa_a3m_filepaths = msa_a3m_filepaths
    infer_cfg.model.data.generate_templates_if_missing = generate_templates_if_missing
    return infer_cfg


@pytest.fixture(scope='function')
def alphafold_model_trainer(alphafold_cfg: DictConfig) -> Tuple[AlphaFold, plt.Trainer]:
    """Setting up the AF model and trainer.

    Args:
        alphafold_cfg (DictConfig): Config object for model and trainer setup.

    Returns:
        Tuple[AlphaFold, plt.Trainer]: AlphaFold model and trainer.
    """
    trainer = setup_trainer(alphafold_cfg, callbacks=[])
    alphafold = AlphaFold.restore_from(
        restore_path=alphafold_cfg.restore_from_path, override_config_path=alphafold_cfg, trainer=trainer
    )
    return alphafold, trainer


@pytest.fixture(scope='function')
def alphafold_model(infer_cfg: DictConfig) -> AlphaFold:
    """Given a config object, setup an AlphaFoldTest object from a checkpoint.

    Args:
        infer_cfg (DictConfig): Config Object containing path to checkpoint.

    Returns:
        AlphaFoldTest: Loaded AlphaFoldTest checkpoint.
    """
    trainer = setup_trainer(infer_cfg, callbacks=[])
    alphafold = AlphaFold.restore_from(
        restore_path=infer_cfg.restore_from_path, override_config_path=infer_cfg, trainer=trainer
    )
    return alphafold


def get_predict_dataset(cfg: DictConfig) -> PredictDataset:
    """Setup of prediction dataset for test purposes; it contains all input features for the
    AlphaFold model, but not the ground truth coordinates.

    Args:
        cfg (DictConfig): Config file to genereate this dataset.

    Returns:
        PredictDataset: dataset object containing AF features for input sequences.
    """
    dataset_paths = get_structured_paths(cfg.model.data)
    dataset = PredictDataset(
        sequences=cfg.sequences,
        seq_names=cfg.seq_names,
        pdb_mmcif_chains_filepath=dataset_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=dataset_paths.mmcif_dicts,
        pdb_obsolete_filepath=dataset_paths.obsolete_filepath,
        template_hhr_filepaths=cfg.model.data.template_hhr_filepaths,
        msa_a3m_filepaths=cfg.model.data.msa_a3m_filepaths,
        generate_templates_if_missing=cfg.model.data.generate_templates_if_missing,
        pdb70_database_path=cfg.model.data.pdb70_database_path,
        cfg=cfg.model,
    )
    return dataset


@pytest.mark.parametrize(
    "outputs",
    [
        ['single', 'msa', 'pair', 'sm_single'],
    ],
)
def test_openfold_prediction_pdb_writer(infer_cfg: DictConfig, outputs: str):
    """Test if OpenFold inference and output writing works with different
        output options (single, msa, pair, sm_single).

    Args:
        infer_cfg (DictConfig): Inference config specifying model options.
        outputs (str): setting determining the output format
    """
    # setup input/output samples for writer
    sequence = infer_cfg.sequences[0]
    seq_name = infer_cfg.seq_names[0]
    N_res = len(sequence)
    batch_size = infer_cfg.model.micro_batch_size
    N_seq = 508

    input_dict = {'seq_name': [seq_name]}
    output_dict = {
        'single': torch.empty(batch_size, N_res, infer_cfg.model.evoformer_stack_config.c_s),
        'msa': torch.empty(batch_size, N_seq, N_res, infer_cfg.model.evoformer_stack_config.c_m),
        'pair': torch.empty(batch_size, N_res, N_res, infer_cfg.model.evoformer_stack_config.c_z),
        'sm_single': torch.empty(batch_size, N_res, infer_cfg.model.structure_module_config.c_s),
    }

    # call writer
    with tempfile.TemporaryDirectory() as temp_dir:
        callback = PredictionFeatureWriter(temp_dir, outputs)
        callback.on_predict_batch_end(
            outputs=output_dict,
            batch=input_dict,
            trainer=None,  # dummy
            pl_module=None,  # dummy
            batch_idx=None,  # dummy
        )


@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    "alphafold_cfg",
    [
        (
            [
                [],
            ]
            * len(MSA_PATHS),
            False,
        ),  # sequence-only inference
        (MSA_PATHS, False),  # sequence-and-msa inference without template
        (MSA_PATHS, True),  # inference with template but no template dataset given
    ],
    indirect=True,
)
def test_openfold_inference_no_output_check(
    alphafold_cfg: DictConfig, alphafold_model_trainer: Tuple[AlphaFold, plt.Trainer]
):
    """Testing if inference itself is running, with no output check.

    Args:
        alphafold_cfg (DictConfig): Config object for the model and dataset setup.
        alphafold_model_trainer (Tuple[AlphaFold, plt.Trainer]): Model and Trainer for inference.
    """
    # setup for inference
    if alphafold_cfg.model.data.generate_templates_if_missing:
        with pytest.raises(ValueError):  # raise error if generate template without template database
            dataset = get_predict_dataset(alphafold_cfg)
    else:
        alphafold_model, trainer = alphafold_model_trainer
        dataset = get_predict_dataset(alphafold_cfg)
        dl = DataLoader(
            dataset,
            batch_size=alphafold_cfg.model.micro_batch_size,
            num_workers=alphafold_cfg.model.data.num_workers,
            collate_fn=collate,
        )
        assert len(dataset) > 0
        # inference
        trainer.predict(alphafold_model, dl, return_predictions=False)


@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    "alphafold_cfg",
    [(MSA_PATHS, False)],
    indirect=True,
)
def test_openfold_inference_lddt_validation_metric_check(alphafold_cfg: DictConfig):
    """Test that checks whether the structure predicted by OpenFold is similar to the ground truth structure.
    For this, the predicted and ground truth coordinates are both represented in atom37 format and fed into
    the `compute_validation_metrics` function that computes the metric `{"lddt_ca"}`.
    [Atom37 format information](https://huggingface.co/spaces/simonduerr/ProteinMPNN/blame/e65166bd70446c6fddcc1581dbc6dac06e7f8dca/alphafold/alphafold/model/all_atom.py)

    In theory we could compute the metrics `{"lddt_ca", "alignment_rmsd", "gdt_ts", "gdt_ha"} via the `compute_validation_metrics` function.
    We do not do this since lddt_ca is the only SE(3)-invariant metrics; all the others rely on alignments.
    The default `superimpose` function used for this in `compute_validation_metrics` is not very good and gives
    therefore incorrect results. Tests of incorporating Kabsch alignment showed that the metric computation itself
    works when a better alignment algorithm is employed, but was not worth the compute requirement here.

    Args:
        alphafold_cfg (DictConfig): Config Object to restore AlphaFold model from checkpoint and initialise dataset.
    """
    # load ground truth data from mmcif files
    mmcif_strings = [load_mmcif_file(mmcif_path) for mmcif_path in CIF_PATHS]
    mmcif_dicts = [parse_mmcif_string(mmcif_string) for mmcif_string in mmcif_strings]
    mmcif_features = [
        create_mmcif_features(mmcif_dict, chain_id) for mmcif_dict, chain_id in zip(mmcif_dicts, CIF_CHAIN_IDS)
    ]
    ground_truth_atom_37_coords_list = [
        torch.from_numpy(mmcif_feature["all_atom_positions"]) for mmcif_feature in mmcif_features
    ]
    # reshape by adding first dimension to match with the batch dimension layout of the predicted coords
    ground_truth_atom_37_coords_list = [
        torch.unsqueeze(gt_coords, 0) for gt_coords in ground_truth_atom_37_coords_list
    ]
    # setup for inference
    trainer = setup_trainer(alphafold_cfg)
    alphafold_model = AlphaFold.restore_from(
        restore_path=alphafold_cfg.restore_from_path, override_config_path=alphafold_cfg, trainer=trainer
    )
    # get prediction dataset from config file, containing only sequences and MSAs, no structures
    dataset = get_predict_dataset(alphafold_cfg)
    data_loader = DataLoader(
        dataset,
        batch_size=alphafold_cfg.model.micro_batch_size,
        num_workers=alphafold_cfg.model.data.num_workers,
        collate_fn=collate,
    )
    # inference
    predicted_structures = trainer.predict(alphafold_model, data_loader, return_predictions=True)
    # get the predicted atom14 coordinates and atom mask telling which positions are valid
    predicted_atom_37_coords_list = [sample["final_atom_positions"] for sample in predicted_structures]
    all_atom_mask_list = [sample["final_atom_mask"] for sample in predicted_structures]
    # compare ground truth to predicted coordinates and calculate lddt_ca metric
    metrics = {"lddt_ca"}
    for predicted_atom_37_coords, ground_truth_atom_37_coord, all_atom_mask in zip(
        predicted_atom_37_coords_list, ground_truth_atom_37_coords_list, all_atom_mask_list
    ):
        validation_metrics = compute_validation_metrics(
            predicted_atom_37_coords, ground_truth_atom_37_coord, all_atom_mask, metrics
        )
        assert validation_metrics["lddt_ca"] > 90
