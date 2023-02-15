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
from bionemo.model.protein.downstream import FineTuneProteinModel, calculate_accuracy, get_data
import numpy as np
import torch

def sec_str_pred_accuracy(output, target, name):
    seq_len = list(target.sum(axis=2).sum(axis=1).cpu().numpy().astype("int"))
    target_seq = [get_data.num2label(label.unsqueeze(0), name) for label in target]
    pred_seq = [get_data.num2label(output.unsqueeze(0), name) for output in output]
    acc = [calculate_accuracy(pred[:l], label[:l]) for (l, pred, label) in zip(seq_len, pred_seq, target_seq)]
    return torch.tensor(np.mean(acc), device="cuda")

def three_state_accuracy(output, target):
    return sec_str_pred_accuracy(output[0], target[0], name="three_state")

def eight_state_accuracy(output, target):
    return sec_str_pred_accuracy(output[1], target[1], name="eight_state")


#@hydra_runner(config_path="../esm1nv/conf", config_name="finetune_config") # ESM
@hydra_runner(config_path="../prott5nv/conf", config_name="finetune_config") # ProtT5
def main(cfg) -> None:

    logging.info("\n\n************* Finetune config ****************")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = setup_trainer(cfg, builder=None)

    model = FineTuneProteinModel(cfg, trainer)
    metrics = {"three_state_accuracy": three_state_accuracy, "eight_state_accuracy": eight_state_accuracy}
    model.add_metrics(metrics=metrics)
    trainer.fit(model)

    if cfg.do_testing:
        if "test_ds" in cfg.model.data:
            trainer.test(model)
        else:
            raise UserWarning("Skipping testing, test dataset file was not provided. Please specify 'test_ds.data_file' in yaml config")

if __name__ == '__main__':
    main()