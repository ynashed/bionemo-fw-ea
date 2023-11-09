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

from functools import partial

from bionemo.data.diffdock.confidence_dataset import ListDataset, diffdock_confidence_dataset
from bionemo.data.diffdock.pdbbind import diffdock_build_dataset
from bionemo.model.molecule.diffdock.setup_trainer import DiffDockModelInference
from bionemo.model.molecule.diffdock.utils.diffusion import t_to_sigma as t_to_sigma_compl


class Singleton(type):
    """
    Ensures single instance of a class.

    Example Usage:
        class MySingleton(metaclass=Singleton)
            pass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DataManager(metaclass=Singleton):
    def __init__(self, cfg, **kwargs) -> None:
        self.t_to_sigma = partial(t_to_sigma_compl, cfg=cfg.model)

        if cfg.model.confidence_mode:
            self.train_ds = diffdock_confidence_dataset(cfg.model.train_ds, mode="train")
            self.validation_ds = diffdock_confidence_dataset(cfg.model.validation_ds, mode="validation")
            self.test_ds = diffdock_confidence_dataset(cfg.model.test_ds, mode="test")

            if not self.train_ds.complex_graphs_ready:
                self.train_ds.build_complex_graphs()
            if not self.validation_ds.complex_graphs_ready:
                self.validation_ds.build_complex_graphs()
            if not self.test_ds.complex_graphs_ready:
                self.test_ds.build_complex_graphs()

            if not (
                self.train_ds.confidence_dataset_ready
                and self.validation_ds.confidence_dataset_ready
                and self.test_ds.confidence_dataset_ready
            ):
                score_model = DiffDockModelInference(cfg.score_infer)
                score_model.eval()

            if not self.train_ds.confidence_dataset_ready:
                self.train_ds.build_confidence_dataset(score_model)
            if not self.validation_ds.confidence_dataset_ready:
                self.validation_ds.build_confidence_dataset(score_model)
            if not self.test_ds.confidence_dataset_ready:
                self.test_ds.build_confidence_dataset(score_model)

            self.train_ds = ListDataset(self.train_ds)
            self.validation_ds = ListDataset(self.validation_ds)
            self.test_ds = ListDataset(self.test_ds)

        else:
            self.train_ds = diffdock_build_dataset(
                cfg.model.train_ds, self.t_to_sigma, _num_conformers=True, mode="train"
            )
            self.validation_ds = diffdock_build_dataset(
                cfg.model.validation_ds, self.t_to_sigma, _num_conformers=False, mode="validation"
            )
            _test_ds = diffdock_build_dataset(cfg.model.test_ds, self.t_to_sigma, _num_conformers=False, mode="test")
            inference_samples = _test_ds.heterograph_store[
                : cfg.model.num_denoising_inference_complexes : cfg.data.world_size
            ]
            self.test_ds = ListDataset(inference_samples)
