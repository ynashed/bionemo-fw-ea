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

three_state_label2num = {"C": 0, "H": 1, "E": 2}
eight_state_label2num = {"_": 0, "E": 1, "G": 2, "T": 3, "H": 4, "S": 5, "B": 6, "I": 7}
resolved_label2num = {"0": 0, "1": 1}

three_state_num2label = {0: "C", 1: "H", 2: "E"}
eight_state_num2label = {0: "_", 1: "E", 2: "G", 3: "T", 4: "H", 5: "S", 6: "B", 7: "I"}
resolved_num2label = {0: "0", 1: "1"}

import copy
import torch
from pathlib import Path
from typing import List, Optional
from pytorch_lightning import Trainer
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils import logging
from torch.utils.data import DataLoader

from bionemo.model.protein.downstream import SSDataModule, SSDataset, get_data
from nemo.utils.model_utils import import_class_by_path

from bionemo.model.protein.downstream import sec_str_pred_model as sspred

from pytorch_lightning.callbacks import Callback

from bionemo.model.core import MLPLightningModule, MLPModel
from bionemo.model.molecule.megamolbart import MegaMolBARTInference
from bionemo.data import SingleValuePredictionDataModule

# TODO: need to test validation in the loop
  
class SSPredictionCallback(Callback):
    def __init__(self, dset_dict, parent_cfg, plugins: Optional[List] = None):
        super().__init__()
        self.dset_dict = dset_dict
        self.plugins = plugins
        self.cfg = parent_cfg
        valid_cfg = [c for c in self.cfg.model.validation.datasets if c['name'] == "SSPred"]
        if len(valid_cfg) != 1:
            raise ValueError("Validation config for SSPred wasn't provided or more than one config was provided")
        self.valid_cfg = valid_cfg[0]
    
    def on_validation_epoch_end(self, trainer, main_model):
        # TODO: replace with prepare and release from inference
        #main_model.freeze()
        # FIXME: find out how to get rid of model.half() here
        torch.manual_seed(self.valid_cfg.random_seed)
        if self.valid_cfg.emb_type == "ESM":
            main_model.half()
            post_process = main_model.model.post_process
            main_model.model.post_process = False
        infer_class = import_class_by_path(self.valid_cfg.infer_target)
        inference_wrapper = infer_class(self.cfg, main_model)
        new_cfg = copy.deepcopy(self.cfg.model)
        new_cfg.data = copy.deepcopy(self.valid_cfg)
        ss_data_module = SSDataModule(
            new_cfg, trainer, model=inference_wrapper
        )
        train_dataset = ss_data_module.get_sampled_train_dataset()
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=self.valid_cfg.batch_size, 
                                      shuffle=False)
        test_dataset = ss_data_module.get_sampled_test_dataset()
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=self.valid_cfg.batch_size, 
                                     shuffle=False)
        results = {}
        # For multi-node training it's important to process data first using all processes
        # before starting to train the SS prediction model on rank 0
        if is_global_rank_zero():
            results = sspred.main(self.valid_cfg, 
                                  train_dataset.data, 
                                  train_dataloader, 
                                  test_dataset.data, 
                                  test_dataloader
                                  )
        torch.distributed.barrier()
        self.log_dict(results, rank_zero_only=True)
        # TODO: replace with release_from_inference
        main_model.unfreeze()
        if self.valid_cfg.emb_type == "ESM":
            main_model.model.post_process = post_process
            main_model.float()


class MLPValidationCallback(Callback):
    def __init__(self, dset_dict, parent_cfg, plugins: Optional[List] = None):
        """
        Callback that can be passed into a model's trainer. At the start of each validation epoch,
        an MLP will be fit to the given benchmarking dataset.
        
        Params
            dset_dict: Dictionary that defines the configuration of this MLP. 
                Ex:
                    {
                        name: ESOL
                        class: MLPValidationCallback
                        enabled: True
                        csv_path: '/data/cheminformatics/benchmark/csv_data/benchmark_MoleculeNet_ESOL.csv'
                        smis_column: 'SMILES'
                        target_column: 'measured log solubility in mols per litre'
                        num_epochs: 50
                        test_fraction: 0.25
                        batch_size: 32
                        learning_rate: 0.001
                        loss_func: MSELoss
                        optimizer: Adam
                        check_val_every_n_epoch: 10
                    }
            parent_cfg: Config dict for the model to which this Callback is being attached
            plugins: Optional plugins for pytorch_lightning.Trainer, currently not used
        """
        super().__init__()
        self.dset_dict = dset_dict
        self.plugins = plugins
        self.cfg = parent_cfg
                    
    def on_validation_epoch_start(self, trainer, main_model):
        main_model.freeze()
        
        inference_wrapper = MegaMolBARTInference(self.cfg, model=main_model)
        
        data = SingleValuePredictionDataModule(Path(self.dset_dict['csv_path']),
                                            self.dset_dict['name'],
                                            inference_wrapper,
                                            self.dset_dict['batch_size'],
                                            self.dset_dict['test_fraction'],
                                            self.dset_dict['smis_column'],
                                            self.dset_dict['target_column'])
        
        mlp = MLPModel()
        mlp_lm = MLPLightningModule(self.dset_dict['name'],
                                    mlp,
                                    self.dset_dict['loss_func'],
                                    self.dset_dict['optimizer'],
                                    self.dset_dict['learning_rate'])
        
        tr = Trainer(max_epochs=self.dset_dict['num_epochs'], 
                        check_val_every_n_epoch=self.dset_dict['check_val_every_n_epoch'], 
                        accelerator='ddp', 
                        gpus=list(range(torch.cuda.device_count())))
                
        tr.fit(mlp_lm, datamodule=data)
        main_model.unfreeze()
