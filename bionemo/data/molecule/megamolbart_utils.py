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

from typing import List
import os
from copy import deepcopy
from omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.utils import logging
from bionemo.data.molecule import MoleculeCsvDataset
from bionemo.data.utils import expand_dataset_paths, check_paths_exist


def _build_train_valid_test_datasets(
    cfg: DictConfig,
    trainer: Trainer,
    num_samples: int,
    filepath: str,
    metadata_path: str,
):
    cfg = deepcopy(cfg)
    with open_dict(cfg):
        cfg['metadata_path'] = metadata_path

    # Get datasets and load data
    logging.info(f'Loading data from {filepath}')
    dataset_paths = expand_dataset_paths(filepath, ".csv")
    errors = check_paths_exist(dataset_paths)
    assert len(errors) == 0, "Following files do not exist %s" % ' '.join(errors)
    logging.info(f'Loading data from {dataset_paths}')
    dataset = MoleculeCsvDataset(dataset_paths=dataset_paths,  cfg=cfg, num_samples=num_samples)
    return dataset


def megamolbart_build_train_valid_test_datasets(
    cfg: DictConfig,
    trainer: Trainer,
    train_valid_test_num_samples: List[int]
):
     # TODO metadata_file is currently not used
    cfg = deepcopy(cfg)
    with open_dict(cfg):
        dataset_path = cfg.pop('dataset_path', '')
        # dataset = cfg.pop('dataset')
        metadata_file = cfg.pop('metadata_file', None)

        ds_train = cfg.dataset.train
        ds_val = cfg.dataset.val
        ds_test = cfg.dataset.test

        cfg.pop('dataset')

    # Build individual datasets.
    if train_valid_test_num_samples[0] > 0:
        filepath = os.path.join(dataset_path, 'train', ds_train)
        metadata_path = os.path.join(dataset_path, 'train', metadata_file) if metadata_file else None
        train_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[0],
                                                        filepath, metadata_path)
    else:
        train_dataset = None

    if train_valid_test_num_samples[1] > 0:
        filepath = os.path.join(dataset_path, 'val', ds_val)
        metadata_path = os.path.join(dataset_path, 'val', metadata_file) if metadata_file else None
        validation_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[1],
                                                            filepath, metadata_path)
    else:
        validation_dataset = None

    if train_valid_test_num_samples[2] > 0:
        filepath = os.path.join(dataset_path, 'test', ds_test)
        metadata_path = os.path.join(dataset_path, 'test', metadata_file) if metadata_file else None
        test_dataset = _build_train_valid_test_datasets(cfg, trainer, train_valid_test_num_samples[2],
                                                        filepath, metadata_path)
    else:
        test_dataset = None

    return (train_dataset, validation_dataset, test_dataset)
