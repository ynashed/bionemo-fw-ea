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

from bionemo.data.preprocess.molecule.uspto50k_preprocess import USPTO50KPreprocess
from bionemo.model.molecule.megamolbart import MegaMolBARTRetroModel
from bionemo.model.utils import get_from_decoder_encoder_or_model, setup_trainer
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector


@hydra_runner(config_path="conf", config_name="downstream_retro_uspto50k")
def main(cfg):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if cfg.do_training:
        trainer = setup_trainer(cfg)

        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            # this method restores state dict from the finetuninig's checkpoint if it is available
            # otherwise loads weights from the "restore_path" model. Also, it overwrites the pretrained model
            # config by the finetuninig config
            model = MegaMolBARTRetroModel.restore_from(
                restore_path=cfg.restore_from_path,
                trainer=trainer,
                save_restore_connector=BioNeMoSaveRestoreConnector(),
                override_config_path=cfg,
            )
        else:
            model = MegaMolBARTRetroModel(cfg.model, trainer)

        assert model._cfg.precision == cfg.trainer.precision
        assert model._cfg.data.max_seq_length == cfg.model.seq_length
        assert get_from_decoder_encoder_or_model(model._cfg, "hidden_dropout") == get_from_decoder_encoder_or_model(
            cfg.model, "hidden_dropout"
        )
        assert model._cfg.data.encoder_augment == cfg.model.data.encoder_augment
        assert model._cfg.data.decoder_augment == cfg.model.data.decoder_augment
        # double check that masking is disabled
        assert not model._cfg.data.encoder_mask
        assert not model._cfg.data.decoder_mask

        logging.info("************** Model parameters and their sizes ***********")
        for name, param in model.named_parameters():
            logging.info(f'{name}: {param.size()}')
            logging.info("***********************************************************")

        logging.info("cfg.************ Starting Training ***********")
        trainer.fit(model)
        logging.info("************** Finished Training ***********")

        if cfg.do_testing:
            logging.info("************** Starting Testing ***********")
            trainer.test(model)
            logging.info("************** Finished Testing ***********")

    else:
        logging.info("************** Starting Data PreProcessing ***********")
        logging.info("Processing data into CSV files")
        data_preprocessor = USPTO50KPreprocess(
            max_smiles_length=cfg.model.data.max_seq_length, data_dir=cfg.model.data.dataset_path
        )

        data_preprocessor.prepare_dataset(
            ngc_registry_target=cfg.model.data.ngc_registry_target,
            ngc_registry_version=cfg.model.data.ngc_registry_version,
            force=True,
        )
        logging.info("************** Finished Data PreProcessing ***********")


if __name__ == '__main__':
    main()
