# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.callbacks.utils import setup_dwnstr_task_validation_callbacks
from bionemo.data import Zinc15Preprocess
from bionemo.model.molecule.molmim import MolMIMModel
from bionemo.model.utils import setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector


@hydra_runner(config_path="conf", config_name="pretrain_xsmall_canonicalized")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    callbacks = setup_dwnstr_task_validation_callbacks(cfg)

    trainer = setup_trainer(cfg, callbacks=callbacks)

    logging.info("************** Model parameters and their sizes ***********")

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = MolMIMModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer, save_restore_connector=BioNeMoSaveRestoreConnector()
            )
        else:
            model = MolMIMModel(cfg.model, trainer)

        trainer.fit(model)

        for name, param in model.named_parameters():
            logging.info(f'{name}: {param.size()}')
            logging.info("***********************************************************")

    else:
        logging.info("************** Starting Data PreProcessing ***********")
        preproc = Zinc15Preprocess(root_directory=cfg.model.data.dataset_path)
        preproc.prepare_dataset(
            links_file=cfg.model.data.links_file,
            max_smiles_length=cfg.model.seq_length,
            output_dir=cfg.model.data.dataset_path,
        )

    if cfg.do_testing:
        logging.info("************** Starting Testing ***********")
        trainer.test(model)


if __name__ == '__main__':
    main()
