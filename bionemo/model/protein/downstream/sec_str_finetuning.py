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

import torch
from functools import lru_cache
from bionemo.model.core import ConvNet, PerTokenMaskedCrossEntropyLoss
from bionemo.model.core.encoder_finetuning import EncoderFineTuning
from nemo.utils.model_utils import import_class_by_path
from bionemo.data.datasets.per_token_value_dataset import PerTokenValueDataModule, PerTokenValueDataset


class FineTuneProteinModel(EncoderFineTuning):

    def __init__(self, cfg, trainer):
        self.full_cfg = cfg
        self.encoder_frozen = self.full_cfg.model.encoder_frozen
        super().__init__(cfg.model, trainer=trainer) 
    
    def configure_optimizers(self):
        super().setup_optimization(optim_config=self.cfg.finetuning_optim)

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def build_loss_fn(self):
        return PerTokenMaskedCrossEntropyLoss()

    def build_task_head(self):
        task_head = ConvNet(self.full_cfg.model.hidden_size, 
                            output_sizes=self.cfg.data.labels_size)
        return task_head

    def setup_encoder_model(self, cfg, trainer):
        infer_class = import_class_by_path(self.full_cfg.infer_target)
        pretrained_model = infer_class(
            self.full_cfg, 
            freeze=self.encoder_frozen, 
            restore_path=self.full_cfg.restore_from_path,
            training=not self.cfg.encoder_frozen,
            adjust_config=False
            )
        return pretrained_model

    @lru_cache
    def data_setup(self):
        if self.encoder_frozen:
            model = self.encoder_model
        else:
            model = None
        self.data_module = PerTokenValueDataModule(
            self.cfg, self.trainer, model=model
        )

    def on_fit_start(self):
        self.build_train_valid_test_datasets()
        return super().on_fit_start()

    def build_train_valid_test_datasets(self):
        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()
        self._test_ds = self.data_module.get_sampled_test_dataset()

    def encoder_forward(self, protein_model, batch: dict):
        if self.encoder_frozen:
            enc_output = batch["embeddings"]
        else:
            enc_output, _ = protein_model.seq_to_hiddens(batch["embeddings"])
            batch_size, seq_len, emb_dim = enc_output.size()
            enc_output = torch.cat([enc_output, torch.zeros((batch_size, (self.full_cfg.model.seq_length - seq_len), emb_dim)).to(device=enc_output.device)], dim=1)

        return enc_output

    def extract_for_task_head(self, input_tensor):
        #NOTE investigate using mixed precision to remove need for float casting; maybe use setup_trainer method
        return input_tensor.float()
  
    def get_target_from_batch(self, batch):
        _, (labels, masks) = PerTokenValueDataset.prepare_batch(batch, self._train_ds)
        padded_labels = []
        padded_masks = []
        for i in range(len(labels)):
            label = labels[i]
            mask = masks[i]
            batch_size, seq_len, n_labels = label.size()
            label = torch.cat([label, torch.zeros((batch_size, (self.full_cfg.model.seq_length - seq_len), n_labels)).to(device=label.device)], dim=1)
            mask = torch.cat([mask, torch.zeros((batch_size, (self.full_cfg.model.seq_length - seq_len))).to(device=mask.device)], dim=1)
            padded_labels.append(label)
            padded_masks.append(mask)
        return (padded_labels, padded_masks)

    def _calc_step(self, batch, batch_idx):
        output_tensor = self.forward(batch)
        target = self.get_target_from_batch(batch)
        loss = self.loss_fn(output_tensor, target)
        return loss, output_tensor, target
