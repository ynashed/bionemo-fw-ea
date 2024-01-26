# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Entry point to DiffDock: training of score/confidence model.

modify parameters from conf/*.yaml
"""
import os
import warnings

from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from rdkit import RDLogger

from bionemo.data.diffdock.data_manager import DataManager
from bionemo.data.diffdock.embedding_preprocess import prep_embedding
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModel as CGScoreModel,
)
from bionemo.model.molecule.diffdock.models.nemo_model import (
    DiffdockTensorProductScoreModelAllAtom as AAScoreModel,
)
from bionemo.model.utils import setup_trainer


warnings.filterwarnings("ignore", module="torch_geometric.data.collate")


@hydra_runner(config_path="conf", config_name="train_score")
def main(cfg) -> None:
    """
    This is the main function conducting data preprocessing, model training
    for both score model and confidence model, and final model inference.
    """
    logging.info("\n\n************** Experiment Configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    if not cfg.model.confidence_mode:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    if cfg.get('seed', None) is not None:
        seed_everything(cfg.seed)

    if cfg.do_embedding_preprocessing:
        logging.info("************** Starting Embedding PreProcessing ***********")
        prep_embedding(cfg.training_data)

    data_manager = DataManager(cfg)
    if cfg.do_preprocessing:
        logging.info("************** Starting Complex Graph PreProcessing ***********")
        data_manager.preprocess()

    if cfg.do_training and data_manager.datasets_ready:
        trainer = setup_trainer(cfg)

        if "all_atoms" in cfg.data and cfg.data.all_atoms:
            model = AAScoreModel(cfg=cfg, trainer=trainer, data_manager=data_manager)
        else:
            model = CGScoreModel(cfg=cfg, trainer=trainer, data_manager=data_manager)

        if cfg.model.confidence_mode:
            logging.info(">>> Configured Confidence Model ...")
        else:
            logging.info(">>> Configured Score Model ...")

        logging.info("************** Starting Training ***********")
        trainer.fit(model)


if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')
    main()
