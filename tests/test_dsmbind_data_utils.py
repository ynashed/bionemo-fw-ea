# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

from torch.utils.data import DataLoader

from bionemo.data.dsmbind.dataset import DSMBindDataset
from bionemo.utils.hydra import load_model_config

from .inference_shared_test_code import get_config_dir


def dsmbind_test_data_path(bionemo_home) -> str:
    path = bionemo_home / "tests" / "dsmbind_test_data" / "test_complexes.pkl"
    return str(path)


def test_dsmbind_dataset(bionemo_home: Path):
    config_path = get_config_dir(bionemo_home, "dsmbind")
    cfg = load_model_config(config_name="train", config_path=config_path)
    processed_data_path = dsmbind_test_data_path(bionemo_home)
    dataset = DSMBindDataset(
        processed_data_path=processed_data_path,
        aa_size=cfg.model.aa_size,
        max_residue_atoms=cfg.model.max_residue_atoms,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, collate_fn=dataset.pl_collate_fn, shuffle=False)
    data_iterator = iter(dataloader)
    batched_binder, batched_target = next(
        data_iterator
    )  # first batch, where the maximum number of ligand atoms is 25 and the number of residues to be considered as in pocket is 50.
    assert batched_binder[0].shape == (cfg.train.batch_size, 25, cfg.model.max_residue_atoms, 3)
    assert len(batched_binder[1]) == cfg.train.batch_size
    assert batched_binder[2].shape == (cfg.train.batch_size, 25, cfg.model.max_residue_atoms)
    assert batched_target[0].shape == (cfg.train.batch_size, 50, cfg.model.max_residue_atoms, 3)
    assert batched_target[1].shape == (cfg.train.batch_size, 50, cfg.model.aa_size)
    assert batched_target[2].shape == (cfg.train.batch_size, 50, cfg.model.max_residue_atoms)
