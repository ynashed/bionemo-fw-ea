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

from bionemo.data import FLIPPreprocess
from bionemo.data.metrics import accuracy, mse, per_token_accuracy
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.model.utils import (
    setup_trainer,
)


@hydra_runner(config_path="../esm1nv/conf", config_name="downstream_flip_sec_str")  # ESM
# @hydra_runner(config_path="../prott5nv/conf", config_name="downstream_flip_sec_str") # ProtT5
def main(cfg) -> None:
    logging.info("\n\n************* Finetune config ****************")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer = setup_trainer(cfg, builder=None)
        model = FineTuneProteinModel(cfg, trainer)
        metrics = {}
        metrics_args = {}
        for idx, name in enumerate(cfg.model.data.target_column):
            if cfg.model.data.task_type == "token-level-classification":
                metrics[name + "_accuracy"] = per_token_accuracy
                metrics_args[name + "_accuracy"] = {"label_id": idx}
            elif cfg.model.data.task_type == "classification":
                metrics[name + "_accuracy"] = accuracy
                metrics_args[name + "_accuracy"] = {}
            elif cfg.model.data.task_type == "regression":
                metrics[name + "_MSE"] = mse
                metrics_args[name + "_MSE"] = {}

        model.add_metrics(metrics=metrics, metrics_args=metrics_args)
        trainer.fit(model)
        logging.info("************** Finished Training ***********")
        if cfg.do_testing:
            logging.info("************** Starting Testing ***********")
            if "test" in cfg.model.data.dataset:
                trainer.test(model)
            else:
                raise UserWarning(
                    "Skipping testing, test dataset file was not provided. Please specify 'dataset.test' in yaml config"
                )
            logging.info("************** Finished Testing ***********")
    else:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = FLIPPreprocess()
        preprocessor.prepare_all_datasets(output_dir=cfg.model.data.preprocessed_data_path)


if __name__ == '__main__':
    main()
