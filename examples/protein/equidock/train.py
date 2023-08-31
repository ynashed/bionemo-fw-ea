#!/bin/bash

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

"""
Entry point to EquiDock.

modify parameters from conf/*.yaml

"""
from bionemo.model.utils import setup_trainer
from bionemo.data.equidock import DataManager
from bionemo.model.protein.equidock.nemo_model import EquiDock
from nemo.utils import logging
from nemo.core.config import hydra_runner
from omegaconf import OmegaConf
from torch.multiprocessing import set_start_method
import torch
import os
os.environ['DGLBACKEND'] = 'pytorch'


torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra_runner(config_path="conf", config_name="train")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    logging.info(f"\n Dataset name {cfg.data.data_name}")

    if cfg.data.num_workers != 0:
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            logging.info(
                "Torch multiprocessing failed with spawn, set num_workers=0!")
            raise RuntimeError(
                f"Torch multiprocessing failed with spawn, set num_workers=0, but it is {cfg.data.num_workers}")

    data_manager = DataManager(cfg)
    cfg.model.input_edge_feats_dim = data_manager.train_ds[0][0].edata['he'].shape[1]
    logging.info(f"\n Dataset name {cfg.data.data_name} loaded!")

    trainer = setup_trainer(cfg)
    model = EquiDock(cfg=cfg, trainer=trainer, data_manager=data_manager)

    if cfg.do_training:
        logging.info(f"***** Training dataset {cfg.data.data_name} ***** ")
        trainer.fit(model=model)

    if cfg.do_testing:
        logging.info(f"***** Testing dataset {cfg.data.data_name} ***** ")
        model = model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)
        model.reload_nemo_model(cfg, trainer, data_manager)
        trainer.test(model, verbose=True)


if __name__ == "__main__":
    main()
