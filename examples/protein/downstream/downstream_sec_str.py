# Copyright (c) 2023, NVIDIA CORPORATION.
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

from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from bionemo.model.utils import (
    setup_trainer,
)
from bionemo.data import FLIPPreprocess
from bionemo.model.protein.downstream import FineTuneProteinModel
from bionemo.data.metrics import per_token_accuracy


@hydra_runner(config_path="../esm1nv/conf", config_name="finetune_config") # ESM
#@hydra_runner(config_path="../prott5nv/conf", config_name="finetune_config") # ProtT5
def main(cfg) -> None:

    logging.info("\n\n************* Finetune config ****************")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    
    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer = setup_trainer(cfg, builder=None)
        model = FineTuneProteinModel(cfg, trainer)
        metrics = {}
        metrics_args = {}
        for idx, name in enumerate(cfg.model.data.labels_col):
            metrics[name + "_accuracy"] = per_token_accuracy
            metrics_args[name + "_accuracy"] = {"label_id": idx}

        model.add_metrics(metrics=metrics, metrics_args=metrics_args)
        trainer.fit(model)
        logging.info("************** Finished Training ***********")
        if cfg.do_testing:
            logging.info("************** Starting Testing ***********")
            if "test" in cfg.model.data.dataset:
                trainer.test(model)
            else:
                raise UserWarning("Skipping testing, test dataset file was not provided. Please specify 'dataset.test' in yaml config")
            logging.info("************** Finished Testing ***********")
    else:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = FLIPPreprocess()
        preprocessor.prepare_dataset(output_dir=cfg.model.data.dataset_path, 
                                     task_name=cfg.model.data.task_name)


if __name__ == '__main__':
    main()