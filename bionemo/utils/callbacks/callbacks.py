from pathlib import Path
from typing import List, Optional

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from bionemo.model.core import MLPLightningModule, MLPModel
from bionemo.model.molecule.megamolbart import MegaMolBARTValidationInferenceWrapper
from bionemo.data import SingleValuePredictionDataModule

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
        
        inference_wrapper = MegaMolBARTValidationInferenceWrapper(main_model, self.cfg)
        
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
