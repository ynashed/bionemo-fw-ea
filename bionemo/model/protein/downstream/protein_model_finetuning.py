# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import lru_cache

import torch
from nemo.utils.model_utils import import_class_by_path

import bionemo.utils
from bionemo.data.datasets import (
    PerTokenValueDataModule,
    PerTokenValueDataset,
    SingleValueDataModule,
    SingleValueDataset,
)
from bionemo.model.core import ConvNet, MLPModel, PerTokenMaskedCrossEntropyLoss
from bionemo.model.core.encoder_finetuning import EncoderFineTuning


class FineTuneProteinModel(EncoderFineTuning):
    def __init__(self, cfg, trainer):
        self.full_cfg = cfg
        self.encoder_frozen = self.full_cfg.model.encoder_frozen
        self.task_type = self.full_cfg.model.data.task_type

        if self.encoder_frozen and self.full_cfg.get('use_peft', False):
            raise ValueError("Using PEFT requires encoder_frozen: False for training")

        super().__init__(cfg.model, trainer=trainer)

    def configure_optimizers(self):
        super().setup_optimization(optim_config=self.cfg.finetuning_optim)

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def build_loss_fn(self):
        if self.task_type in ['regression', 'classification']:
            loss = bionemo.utils.lookup_or_use(torch.nn, self.cfg.loss_func)
        elif self.task_type == 'token-level-classification':
            loss = PerTokenMaskedCrossEntropyLoss()
        return loss

    def build_task_head(self):
        if self.task_type in ['regression', 'classification']:
            if len(self.cfg.data.target_sizes) != 1:
                raise ValueError("Classification and regression tasks do not support multi-head predictions")
            task_head = MLPModel(
                layer_sizes=[
                    self.encoder_model.cfg.model.hidden_size,
                    self.cfg.hidden_layer_size,
                    self.cfg.data.target_sizes[0],
                ],
                dropout=self.cfg.dropout_rate,
            )

        elif self.task_type == 'token-level-classification':
            task_head = ConvNet(
                self.full_cfg.model.hidden_size,
                output_sizes=self.cfg.data.target_sizes,
                bottleneck_dim=self.full_cfg.model.cnn_dim,
                dropout_rate=self.full_cfg.model.dropout_rate,
            )
        return task_head

    def setup_encoder_model(self, cfg, trainer):
        # Uses PEFT flag to determine whether to load enconder +/- adapters
        infer_class = import_class_by_path(self.full_cfg.infer_target)
        pretrained_model = infer_class(
            self.full_cfg,
            freeze=self.encoder_frozen,
            restore_path=self.full_cfg.restore_from_path,
            training=not self.cfg.encoder_frozen,
            adjust_config=False,
        )
        return pretrained_model

    @lru_cache
    def data_setup(self):
        # TODO: address behavior for when we want peft
        if self.encoder_frozen:
            model = self.encoder_model
        else:
            model = None

        if self.task_type in ['regression', 'classification']:
            self.data_module = SingleValueDataModule(self.cfg, self.trainer, model=model)
        elif self.task_type == 'token-level-classification':
            self.data_module = PerTokenValueDataModule(self.cfg, self.trainer, model=model)

    def on_fit_start(self):
        self.build_train_valid_test_datasets()
        return super().on_fit_start()

    def build_train_valid_test_datasets(self):
        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()
        self._test_ds = self.data_module.get_sampled_test_dataset()

    def encoder_forward(self, protein_model, batch: dict):
        '''
        Params:
            batch: Dictionary that conatains
                "embeddings": this is a sequence of amino acids when `encoder_frozen` is False.
        '''
        if self.encoder_frozen:
            # If encoder is frozen, the Dataset returns the embedding directly
            enc_output = batch["embeddings"]
        elif self.task_type in ['regression', 'classification']:
            # contains a sequence, not an embedding
            enc_output = protein_model.seq_to_embeddings(batch["embeddings"])  # (B, D)
        else:
            # contains a sequence, not an embedding
            enc_output, _ = protein_model.seq_to_hiddens(batch["embeddings"])  # (B, S, D)
            batch_size, seq_len, emb_dim = enc_output.size()
            enc_output = torch.cat(
                [
                    enc_output,
                    torch.zeros((batch_size, (self.full_cfg.model.seq_length - seq_len), emb_dim)).to(
                        device=enc_output.device
                    ),
                ],
                dim=1,
            )

        return enc_output

    def extract_for_task_head(self, input_tensor):
        # NOTE investigate using mixed precision to remove need for float casting; maybe use setup_trainer method
        return input_tensor.float()

    def get_target_from_batch(self, batch):
        _, (labels, masks) = PerTokenValueDataset.prepare_batch(batch, self._train_ds)
        padded_labels = []
        padded_masks = []
        for i in range(len(labels)):
            label = labels[i]
            mask = masks[i]
            batch_size, seq_len, n_labels = label.size()
            label = torch.cat(
                [
                    label,
                    torch.zeros((batch_size, (self.full_cfg.model.seq_length - seq_len), n_labels)).to(
                        device=label.device
                    ),
                ],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros((batch_size, (self.full_cfg.model.seq_length - seq_len))).to(device=mask.device)],
                dim=1,
            )
            padded_labels.append(label)
            padded_masks.append(mask)
        return (padded_labels, padded_masks)

    def _calc_step(self, batch, batch_idx):
        output_tensor = self.forward(batch)
        if self.task_type in ['regression', 'classification']:
            _, target = SingleValueDataset.prepare_batch(batch, self._train_ds, task=self.task_type)
        elif self.task_type == 'token-level-classification':
            target = self.get_target_from_batch(batch)
        loss = self.loss_fn(output_tensor, target)
        return loss, output_tensor, target

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        custom_state_dict = super(FineTuneProteinModel, self).state_dict()

        if self.full_cfg.get('use_peft', False):  # skipped if use_peft is false or not present in config
            custom_state_dict.update(self.encoder_model.model.state_dict())

        return custom_state_dict
