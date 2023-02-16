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

import torch
from pathlib import Path
from typing import List, Optional
from pytorch_lightning import Trainer
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils import logging
from torch.utils.data import DataLoader

from bionemo.model.protein.downstream import SSDataset, get_data
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
        assert len(valid_cfg) == 1
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
        traindata = get_data(
            datafile=self.valid_cfg.train_ds.data_file, 
            model=inference_wrapper,
            emb_batch_size=self.valid_cfg.emb_batch_size, 
            max_seq_length=self.cfg.model.seq_length
            )
        trainDataset = SSDataset(traindata)
        train_dataloader = DataLoader(trainDataset, batch_size=self.valid_cfg.batch_size, shuffle=False)
        logging.info("SS prediction training dataloader created...")
        test_datafile = self.valid_cfg.test_ds.data_file
        testdata = get_data(
            datafile=test_datafile, 
            model=inference_wrapper,
            emb_batch_size=self.valid_cfg.emb_batch_size,
            max_seq_length=self.cfg.model.seq_length
            )
        test_dataset = SSDataset(testdata)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        logging.info("SS prediction test dataloader created...")
        results = {}
        # For multi-node training it's important to process data first using all processes
        # before starting to train the SS prediction model on rank 0
        if is_global_rank_zero():
            results = sspred.main(self.valid_cfg, traindata, train_dataloader, testdata, test_dataloader)
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
