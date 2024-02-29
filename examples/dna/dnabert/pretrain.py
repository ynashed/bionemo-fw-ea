# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.data.preprocess.dna.preprocess import DNABERTPreprocess
from bionemo.model.dna.dnabert import DNABERTModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector


@hydra_runner(config_path="conf", config_name="dnabert_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    np.random.seed(cfg.model.seed)
    pl.seed_everything(cfg.model.seed)

    if cfg.do_training:
        trainer = setup_trainer(cfg)
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = DNABERTModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=BioNeMoSaveRestoreConnector()
            )
        else:
            model = DNABERTModel(cfg.model, trainer)
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
        logging.info("*************** Finish Training ************")
    else:
        logging.info("************** Starting Preprocessing ***********")
        logging.warning(
            "For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose."
        )

        preprocessor = DNABERTPreprocess(
            cfg.model.data.dataset_path,
            cfg.model.tokenizer.model,
            cfg.model.tokenizer.vocab_file,
            cfg.model.tokenizer.k,
            cfg.model.data.dataset,
        )
        preprocessor.preprocess()

        logging.info("*************** Finish Preprocessing ************")


if __name__ == '__main__':
    main()
