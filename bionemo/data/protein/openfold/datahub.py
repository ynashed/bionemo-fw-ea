from pathlib import Path
from typing import Dict, Union

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from bionemo.data.protein.openfold.dataloaders import (
    FinetuningDataloader,
    InitialTrainingDataloaderPT,
    ValidationDataloader,
)
from bionemo.data.protein.openfold.dataloaders_pq import InitialTrainingDataloaderPQ
from bionemo.data.protein.openfold.datasets import (
    FinetuningDataset,
    InitialTrainingDataset,
    SelfDistillationDataset,
    ValidationDataset,
)
from bionemo.data.protein.openfold.samplers import FinetuningSampler, InitialTrainingSampler, ValidationSampler
from bionemo.model.protein.openfold.optim_hub import OptimHub


# TODO: dataloaders should inlcude gradient checkpoint as this is probably a stanard across bionemo models


def get_structured_paths(data_cfg) -> DictConfig:
    # data
    # ├── open_protein_set
    # │   └── original  # 1.1 TiB (~1.3M files)
    # │       ├── LICENSE
    # │       ├── duplicate_pdb_chains.txt
    # │       ├── pdb/
    # │       └── uniclust30/
    # │   └── processed  # 1.1 TiB (33 files)
    # │       └── pdb_alignments/
    # │       └── uniclust30_alignments/
    # │       └── uniclust30_targets/
    # └── pdb_mmcif
    #     └── original  # 55 GiB (+200k files)
    #         ├── clusters-by-entity-40.txt
    #         ├── obsolete.dat
    #         └── raw/
    #     └── processed  # +15 GiB (~1k files)
    #         ├── chains.csv
    #         ├── dicts/
    #         ├── dicts_preprocessing_logs.csv
    #         └── obsolete.dat

    data_cfg.dataset_path = Path(data_cfg.dataset_path)
    return OmegaConf.create(
        {
            'mmcif_chains': data_cfg.dataset_path / 'pdb_mmcif' / data_cfg.dataset_variant / 'chains.csv',
            'mmcif_dicts': data_cfg.dataset_path / 'pdb_mmcif' / data_cfg.dataset_variant / 'dicts',
            'obsolete_filepath': data_cfg.dataset_path / 'pdb_mmcif' / data_cfg.dataset_variant / 'obsolete.dat',
            'alignments_dirpath': data_cfg.dataset_path
            / 'open_protein_set'
            / data_cfg.dataset_variant
            / 'pdb_alignments',
            'uniclust30_alignments': data_cfg.dataset_path
            / 'open_protein_set'
            / data_cfg.dataset_variant
            / 'uniclust30_alignments',
            'uniclust30_targets': data_cfg.dataset_path
            / 'open_protein_set'
            / data_cfg.dataset_variant
            / 'uniclust30_targets',
        }
    )


def get_initial_training_dl(
    model_cfg: Union[DictConfig, Dict], train_session_cfg: Union[DictConfig, Dict], ds_cfg: Union[DictConfig, Dict]
) -> DataLoader:
    ds_paths = get_structured_paths(model_cfg.data)

    initial_training_dataset = InitialTrainingDataset(
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        pdb_alignments_dirpath=ds_paths.alignments_dirpath,
        max_pdb_release_date=ds_cfg.train_max_pdb_release_date,
        alphafold_config=model_cfg,
        filter_by_alignments=model_cfg.data.filter_by_alignments,
        realign_when_required=ds_cfg.realign_when_required,
        use_only_pdb_chain_ids=model_cfg.data.use_only_pdb_chain_ids,
        name=f"initial_training_dataset_{train_session_cfg.rank}",  # TODO: is this necessary + replace placeholder
    )

    initial_training_sampler = InitialTrainingSampler(
        dataset=initial_training_dataset,
        device_batch_size=model_cfg.micro_batch_size,
        global_batch_size=model_cfg.global_batch_size,
        num_train_iters=model_cfg.num_train_iters,
        seed=model_cfg.seed,
        is_distributed=True,
        rank=train_session_cfg.rank,
        world_size=train_session_cfg.world_size,
        num_prev_iters=train_session_cfg.iteration,
    )

    # initial_training_dataset = InitialTrainingDataset(
    if OptimHub.config('dataloader_pq'):
        InitialTrainingDataloader = InitialTrainingDataloaderPQ
    else:
        InitialTrainingDataloader = InitialTrainingDataloaderPT

    initial_training_dataloader = InitialTrainingDataloader(
        dataset=initial_training_dataset,
        sampler=initial_training_sampler,
        local_batch_size=model_cfg.micro_batch_size,
        num_workers=ds_cfg.num_workers,
        seed=model_cfg.seed,
        uniform_recycling_iters=list(range(0, model_cfg.num_recycling_iters + 1)),
        num_prev_iters=train_session_cfg.iteration,
        use_threading=ds_cfg.threading_enabled,
        prefetch_factor=2,
    )

    return initial_training_dataloader


def get_validation_dataloader(model_cfg, dist_cfg: Union[DictConfig, Dict], ds_cfg: Union[DictConfig, Dict]):
    ds_paths = get_structured_paths(model_cfg.data)
    validation_dataset = ValidationDataset(
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        pdb_alignments_dirpath=ds_paths.alignments_dirpath,
        min_cameo_submission_date=ds_cfg.val_min_cameo_submission_date,
        max_cameo_submission_date=ds_cfg.val_max_cameo_submission_date,
        max_sequence_length=ds_cfg.val_max_sequence_length,
        alphafold_config=model_cfg,
        filter_by_alignments=model_cfg.data.filter_by_alignments,
        realign_when_required=ds_cfg.realign_when_required,
        use_only_pdb_chain_ids=model_cfg.data.use_only_pdb_chain_ids,
        name=f"validation_dataset_{dist_cfg.rank}",
    )
    validation_sampler = ValidationSampler(
        dataset=validation_dataset,
        is_distributed=True,
        rank=dist_cfg.rank,
        world_size=dist_cfg.world_size,
    )
    validation_dataloader = ValidationDataloader(
        dataset=validation_dataset,
        sampler=validation_sampler,
        num_workers=ds_cfg.num_workers,
    )

    return validation_dataloader


def get_finetuning_dl(
    model_cfg: Union[DictConfig, Dict], train_session_cfg: Union[DictConfig, Dict], ds_cfg: Union[DictConfig, Dict]
):
    ds_paths = get_structured_paths(model_cfg.data)
    initial_training_dataset = InitialTrainingDataset(
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        pdb_alignments_dirpath=ds_paths.alignments_dirpath,
        max_pdb_release_date=ds_cfg.train_max_pdb_release_date,
        alphafold_config=model_cfg,
        filter_by_alignments=model_cfg.data.filter_by_alignments,
        realign_when_required=ds_cfg.realign_when_required,
        use_only_pdb_chain_ids=model_cfg.data.use_only_pdb_chain_ids,
        name=f"initial_training_dataset_{train_session_cfg.rank}",
    )

    self_distillation_dataset = SelfDistillationDataset(
        uniclust30_alignments_dirpath=ds_paths.uniclust30_alignments,
        uniclust30_targets_dirpath=ds_paths.uniclust30_targets,
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        max_pdb_release_date=ds_cfg.train_max_pdb_release_date,
        realign_when_required=ds_cfg.realign_when_required,
        alphafold_config=model_cfg,
        name=f"self_distillation_dataset_{train_session_cfg.rank}",
    )

    finetuning_dataset = FinetuningDataset(
        initial_training_dataset=initial_training_dataset,
        self_distillation_dataset=self_distillation_dataset,
    )

    finetuning_sampler = FinetuningSampler(
        dataset=finetuning_dataset,
        dataset_weights={
            "initial_training": 0.25,
            "self_distillation": 0.75,
        },
        device_batch_size=model_cfg.micro_batch_size,
        global_batch_size=model_cfg.global_batch_size,
        num_train_iters=model_cfg.num_train_iters,
        seed=model_cfg.seed,
        is_distributed=True,
        rank=train_session_cfg.rank,
        world_size=train_session_cfg.world_size,
        num_prev_iters=train_session_cfg.iteration,
    )

    finetuning_dataloader = FinetuningDataloader(
        dataset=finetuning_dataset,
        sampler=finetuning_sampler,
        device_batch_size=model_cfg.micro_batch_size,
        num_workers=ds_cfg.num_workers,
        seed=model_cfg.seed,
        uniform_recycling_iters=list(range(0, model_cfg.num_recycling_iters + 1)),
        num_prev_iters=train_session_cfg.iteration,
    )

    return finetuning_dataloader


def get_training_dataloader(
    model_cfg: Union[DictConfig, Dict], train_session_cfg: Union[DictConfig, Dict], ds_cfg: Union[DictConfig, Dict]
):
    stage = model_cfg.stage
    if stage == "initial_training":
        return get_initial_training_dl(model_cfg=model_cfg, train_session_cfg=train_session_cfg, ds_cfg=ds_cfg)
    elif stage == "finetuning":
        return get_finetuning_dl(model_cfg=model_cfg, train_session_cfg=train_session_cfg, ds_cfg=ds_cfg)
    else:
        # no custom exception since this should be unrecoverable
        raise Exception(f"Stage {stage} is not recognised. Please choose either initial_training or finetuning.")
