from typing import Union, Optional, List
import json

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from nemo.utils import logging
import bionemo.utils

class MLPModel(nn.Module):
    def __init__(self, 
                 layer_sizes: Optional[List[int]] = None, 
                 dropout: float = 0.1, 
                 activation_function: Optional[nn.Module] = None):
        """
        Simple MLP Model for validation on benchmark datasets
        
        Params
            layer_sizes: List of layer sizes. By default: [256, 128, 1]
            dropout: float
            activation_function: PyTorch activation function. Uses ReLU if not provided
        """
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [256, 128, 1]
        self.linear_layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layer_norm = nn.LayerNorm(layer_sizes[0]) 
        self.act = nn.ReLU() if activation_function is None else activation_function
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.linear_layers[:-1]:
            x = self.act(self.dropout(layer(x)))
            
        x = self.linear_layers[-1](x)
        return x

class MLPLightningModule(LightningModule):
    def __init__(self, 
                 dset_name: str, 
                 mlp_model: MLPModel, 
                 loss_func: Union[str, nn.Module], 
                 optimizer: Union[str, torch.optim.Optimizer], 
                 lr=0.001):
        """
        LightningModule capturing training logic for MLPModel
        
        Params
            dset_name: String
            mlp_model: MLPModel instance
            loss_func: String or PyTorch loss function
            optimizer: String or PyTorch optimizer
            lr: Float
        """
        super().__init__()
        self.mlp_model = mlp_model
        self.loss_func = bionemo.utils.lookup_or_use(torch.nn, loss_func)
        self.optim = optimizer
        self.lr = lr
        self.dset_name = dset_name
        
    def forward(self, x):
        return self.mlp_model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = torch.squeeze(self.mlp_model(x))
        loss = self.loss_func(output, y)
        return loss
    
    def on_validation_start(self) -> None:
        self.best_val_loss = float('inf')
        self.best_val_epoch = -1
    
    def on_validation_epoch_start(self) -> None:
        # track validation loss for current epoch
        self.current_val_loss = 0
        self.num_val_steps = 0
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = torch.squeeze(self.mlp_model(x))
        loss = self.loss_func(output, y)
        self.current_val_loss += loss.item()
        self.num_val_steps += 1
        return loss
    
    def on_validation_epoch_end(self) -> None:
        mean_epoch_loss = self.current_val_loss / self.num_val_steps
        
        if mean_epoch_loss < self.best_val_loss:
            self.best_val_loss = mean_epoch_loss
            self.best_val_epoch = self.current_epoch
            
        if self.current_epoch == self.trainer.max_epochs - 1:
            results_dict = {}
            results_dict[f'{self.dset_name}_val_mlp_best-loss'] = self.best_val_loss
            results_dict[f'{self.dset_name}_val_mlp_best-epoch'] = self.best_val_epoch
            self.log_dict(results_dict)
            logging.info(f'\nMLP Validation Results:\n{json.dumps(results_dict, indent=2, default=str)}')
                
    def configure_optimizers(self):
        return bionemo.utils.lookup_or_use(torch.optim, self.optim, self.mlp_model.parameters(), lr=self.lr)
