# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.core.config.schedulers import SchedulerParams
from nemo.core.optim.lr_scheduler import register_scheduler
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.optim.lr_scheduler import LinearLR

from bionemo.data.preprocess.dna.preprocess import BasenjiDatasetPreprocessor
from bionemo.model.dna.enformer import Enformer
from bionemo.model.utils import setup_trainer


@dataclass
class LinearLRParams(SchedulerParams):
    # warning: all params must be explicitly overriden in the config
    # as registered params are not passed down
    start_factor: float = 0.000001
    end_factor: float = 1.0
    total_iters: int = 5000
    last_epoch: int = -1
    verbose: bool = True


@hydra_runner(config_path="conf", config_name="enformer_pretrain")
def main(cfg) -> None:
    register_scheduler(name='LinearLR', scheduler=LinearLR, scheduler_params=LinearLRParams)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing required keys in config:\n{missing_keys}")

    if cfg.do_preprocess:
        preprocessor = BasenjiDatasetPreprocessor(cfg.model.data)
        preprocessor.process()

    np.random.seed(cfg.model.seed)
    pl.seed_everything(cfg.model.seed)
    trainer = setup_trainer(cfg, callbacks=[])

    if cfg.restore_from_path:
        # since fine-tuning has different head configuration, let's override model config
        enformer = Enformer.restore_from(restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer)
        enformer.setup_training_data(cfg.model.train_ds)
        enformer.setup_validation_data(cfg.model.validation_ds)
    else:
        enformer = Enformer(cfg.model, trainer)
    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(enformer)
        logging.info("*************** Finish Training ************")

    if cfg.do_testing:
        enformer.setup_test_data(cfg.model.test_ds)
        trainer.test(enformer)


if __name__ == '__main__':
    main()
