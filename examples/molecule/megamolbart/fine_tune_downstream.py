import os
import torch
import torch.nn as nn
import numpy as np
import bionemo.utils
import pytorch_lightning as ptl
from functools import lru_cache
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict, DictConfig
from bionemo.model.molecule.megamolbart import MegaMolBARTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from bionemo.model.core import MLPModel
from bionemo.model.core.encoder_finetuning import EncoderFineTuning
from bionemo.data.finetune_dataset import FineTuneDataModule
from bionemo.model.utils import (
    setup_trainer,
    restore_model,
)

class FineTuneMegaMolBART(EncoderFineTuning):

    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer=trainer)

        self.batch_target_name = cfg.downstream_task.target_column

        self.cfg = cfg

    def build_loss_fn(self):
        return bionemo.utils.lookup_or_use(torch.nn, self.cfg.downstream_task.loss_func)

    def build_task_head(self):
        regressor = MLPModel(layer_sizes=[self.encoder_model._cfg.hidden_size, self.cfg.downstream_task.hidden_layer_size, self.cfg.downstream_task.n_outputs],
            dropout=0.1,
        )
        task_head = nn.Sequential(regressor, nn.Flatten(start_dim=0))
        return task_head

    def setup_encoder_model(self, cfg, trainer):

        pretrained_model = self.load_model(cfg, trainer)
        pretrained_model.decoder = torch.nn.Identity()

        return pretrained_model

    def load_model(self, model_cfg, trainer):
        """Load saved model checkpoint
        Params:
            checkpoint_path: path to nemo checkpoint
        Returns:
            MegaMolBART trained model
        # """

        if model_cfg.downstream_task.megamolbart_model_path is not None:
            model = MegaMolBARTModel.restore_from(
                restore_path=model_cfg.downstream_task.megamolbart_model_path,
                trainer=trainer,
                save_restore_connector=NLPSaveRestoreConnector(),
            )


        logging.info(f"Encoder weights frozen set to: {model_cfg.downstream_task.freeze_encoder_weights}")
        if model_cfg.downstream_task.freeze_encoder_weights:
            model.freeze()

        return model

    # the lru cache is kind of a hacky way to make sure this isn't set up if
    # it is already initialized, since this function doesn't return anything
    @lru_cache
    def data_setup(self):
        self.data_module = FineTuneDataModule(
            self.cfg, self.encoder_model.tokenizer, self.trainer,
        )

    def on_fit_start(self):
        self.build_train_valid_test_datasets()
        return super().on_fit_start()

    def build_train_valid_test_datasets(self):

        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()

    def encoder_forward(self, bart_model, batch: dict):
        tokens, mask = self.process_batch(batch)
        enc_output = bart_model.encode(tokens_enc=tokens, enc_mask=mask)
        output_tensor = self.hidden_to_embedding(enc_output, mask) 

        return output_tensor

    def extract_for_task_head(self, input_tensor):
        #NOTE investigate using mixed precision to remove need for float casting; maybe use setup_trainer method
        return input_tensor.float()

    def process_batch(self, batch):
        """Build the batch."""

        token_ids = batch['token_ids']
        mask = (token_ids != self.encoder_model.tokenizer.pad_id)

        tk = torch.tensor(token_ids, dtype=torch.int64).cuda()

        mask = torch.tensor(mask, dtype=torch.int64,
                                    device=token_ids.device)
        return tk, mask

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
    
    def get_target_from_batch(self, batch):
        ret = batch['target']

        return ret.float()

@hydra_runner(config_path="conf", config_name="finetune_config") 
def main(cfg) -> None:

    logging.info("\n\n************* Fintune config ****************")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    np.random.seed(cfg.model.downstream_task.seed)
    ptl.seed_everything(cfg.model.downstream_task.seed)

    trainer = setup_trainer(
         cfg, builder=None)

    model = FineTuneMegaMolBART(cfg.model, trainer)
    trainer.fit(model)

if __name__ == '__main__':
    main()
