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


from omegaconf.omegaconf import OmegaConf
from nemo.core.config import hydra_runner
from nemo.utils import logging

from bionemo.model.molecule.megamolbart import MegaMolBARTModel
from bionemo.data import SmilesPreprocess
from bionemo.model.utils import setup_trainer
from bionemo.utils.callbacks.callback_utils import setup_callbacks

@hydra_runner(config_path="conf", config_name="pretrain_xsmall_span_aug")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    callbacks = setup_callbacks(cfg)

    trainer = setup_trainer(cfg, callbacks=callbacks)
    model = MegaMolBARTModel(cfg.model, trainer)

    logging.info("************** Model parameters and their sizes ***********")
    for name, param in model.named_parameters():
        logging.info(f'{name}: {param.size()}')
        logging.info("***********************************************************")

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        trainer.fit(model)
    else:
        logging.info("************** Starting Data PreProcessing ***********")
        SmilesPreprocess().prepare_dataset(links_file=cfg.model.data.links_file,
                                           output_dir=cfg.model.data.dataset_path,
                                           max_smiles_length=cfg.model.seq_length)

    if cfg.do_testing:
        logging.info("************** Starting Testing ***********")
        trainer.test(model)


if __name__ == '__main__':
    main()
