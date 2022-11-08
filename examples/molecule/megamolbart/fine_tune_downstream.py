import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import bionemo.utils
import pytorch_lightning as ptl
from nemo.core.config import hydra_runner
from nemo.utils import logging
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.trainer.trainer import Trainer
from bionemo.model.molecule.megamolbart import MegaMolBARTModel
from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable
from omegaconf.omegaconf import OmegaConf, open_dict, DictConfig
from pathlib import Path
from typing import Union
from nemo.utils.app_state import AppState
from nemo.utils.exp_manager import exp_manager
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin, NLPSaveRestoreConnector
from bionemo.model.core import MLPModel
from bionemo.core import BioNeMoDataModule
from bionemo.data.finetune_dataset import FineTuneDataset
from bionemo.data.finetune_dataset import FineTuneDataModule


class FineTuneMegaMolBART(ModelPT, Exportable):

    def __init__(self, cfg: "DictConfig", trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.cfg = cfg

        self.pretrained_model = self.load_model(cfg)

        #set input layer dims of MLP based on hidden_size from pretrained model
        self.regressor = MLPModel(layer_sizes=[self.pretrained_model._cfg.hidden_size, 128, 1], dropout=0.1)

        self.loss_fn = bionemo.utils.lookup_or_use(torch.nn, cfg.downstream_task.loss_func)

        #check that decoder exists in megamolbart model and following modules
        self.pretrained_model.decoder = torch.nn.Identity()

        self.data_module = FineTuneDataModule(self.cfg, self.pretrained_model.tokenizer)

        self._build_train_valid_datasets()
        self.setup_training_data(self.cfg)
        self.setup_validation_data(self.cfg)

        #megatronbart.token_fc = torch.nn.Identity()
        #megatronbart.loss_fn = torch.nn.Identity()
        #megatronbart.log_softmax = torch.nn.Identity()

    def forward(self, token_ids, mask):
        """ Apply SMILES strings to model
        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.
        Arg:
            token_ids: tensor of token_ids of shape (seq_len, batch_size),
            mask: bool tensor of padded elems of shape (seq_len, batch_size)
        Returns:
            Output from model (dict containing key "token_output")
        """

        enc_output = self.pretrained_model.encode(tokens_enc=token_ids, enc_mask=mask) #return hiddens
        embeddings = self.hidden_to_embedding(enc_output, mask) 

        token_output = self.regressor(embeddings.float())
  
        output = {'token_output': torch.squeeze(token_output)}

        return output

    def load_model(self, model_cfg):
        """Load saved model checkpoint
        Params:
            checkpoint_path: path to nemo checkpoint
        Returns:
            MegaMolBART trained model
        # """

        # trainer required for restoring model parallel models
        trainer = Trainer(
            plugins=NLPDDPPlugin(),
            devices=1,
            accelerator='gpu',
            precision=32, #TODO: Run benchmark to verify this value has no or
            #                     minimum impact on KPIs.
        )

        app_state = AppState()
        if model_cfg.downstream_task.megamolbart_model_path is not None:
            model = MegaMolBARTModel.restore_from(
                restore_path=model_cfg.downstream_task.megamolbart_model_path,
                trainer=trainer,
                save_restore_connector=NLPSaveRestoreConnector(),
            )

        return model
    
    def validation_epoch_end(self, outputs):
        pass

    def hidden_to_embedding(self, enc_output, enc_mask):
        """Computes embedding and padding mask for smiles.
        Params
            enc_output: hidden-state array
            enc_mask:   boolean mask
        Returns
            embeddings
        """

        lengths = enc_mask.sum(dim=1, keepdim=True)
        embeddings = torch.sum(enc_output*enc_mask.unsqueeze(-1), dim=1) / lengths

        return embeddings

    def _calc_step(self, batch, batch_idx):

        tokens, enc_mask = self.process_batch(batch)
        output_tensor = self.forward(tokens, enc_mask)

        target_tokens = batch['target']
        loss = self.loss_fn(output_tensor['token_output'], target_tokens.float())

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self._calc_step(batch, batch_idx)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self._calc_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)

        return loss    
    
    def process_batch(self, batch):
        """Build the batch."""

        token_ids = batch['token_ids']
        mask = (token_ids != self.pretrained_model.tokenizer.pad_id)

        tk = torch.tensor(token_ids, dtype=torch.int64).cuda()

        mask = torch.tensor(mask, dtype=torch.int64,
                                    device=token_ids.device)
        return tk, mask

    def _build_train_valid_datasets(self):

        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()

    def setup_training_data(self, cfg):
        """
        Setups data loader to be used in training

        Args:
            cfg: data layer parameters.
        Returns:

        """

        self._train_dl = DataLoader(self._train_ds, batch_size=cfg.downstream_task.batch_size, drop_last=True)
        self.data_module.adjust_train_dataloader(self, self._train_dl)

    def setup_validation_data(self, cfg):
        """
        Setups data loader to be used in validation
        Args:

            cfg: data layer parameters.
        Returns:

        """

        self._validation_dl = DataLoader(self._validation_ds, batch_size=cfg.downstream_task.batch_size, drop_last=True)
        self.data_module.adjust_val_dataloader(self, self._validation_dl)

    def list_available_models(self):
        return []

@hydra_runner(config_path="conf", config_name="finetune_config") 
def main(cfg) -> None:

    logging.info("\n\n************* Fintune config ****************")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    np.random.seed(cfg.model.downstream_task.seed)
    ptl.seed_everything(cfg.model.downstream_task.seed)

    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=cfg.model.downstream_task.epochs, logger=False)
    exp_manager(trainer, cfg.get("exp_manager", None)) 
    
    #NOTE try use setup_trainer from util.py from dev branch (configure_plugins/configure_callbacks/resume_checkpoint may cause error)
    #trainer = setup_trainer(cfg, builder=FintuneTrainerBuilder())
    #NOTE instantiate encoder outside of finetuning class to allow for flexibility

    model = FineTuneMegaMolBART(cfg.model, trainer)
    trainer.fit(model)

if __name__ == '__main__':
    main()
