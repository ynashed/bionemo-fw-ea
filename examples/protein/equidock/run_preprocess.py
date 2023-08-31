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

from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging

from bionemo.data.equidock.preprocess import preprocess


@hydra_runner(config_path="conf", config_name="train")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")

    logging.info(f"\n Dataset name {cfg.data.data_name}")
    train_cfg = cfg.model.train_ds
    test_cfg = cfg.model.test_ds
    val_cfg = cfg.model.validation_ds

    logging.info(f"\n Preprocessing validation dataset!")
    logging.info(f"\n{OmegaConf.to_yaml(val_cfg)}")
    preprocess(val_cfg)

    logging.info(f"\n Preprocessing testing dataset!")
    logging.info(f"\n{OmegaConf.to_yaml(test_cfg)}")
    preprocess(test_cfg)

    logging.info(f"\n Preprocessing training dataset!")
    logging.info(f"\n{OmegaConf.to_yaml(train_cfg)}")
    preprocess(train_cfg)


if __name__ == "__main__":
    main()
