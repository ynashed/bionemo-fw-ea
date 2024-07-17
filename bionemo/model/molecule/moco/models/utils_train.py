# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import torch
from lightning import pytorch as pl


class ExponentialMovingAverage:
    """from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters."""

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use the number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def to(self, device):
        self.shadow_params = [p.to(device) for p in self.shadow_params]
        self.collected_params = [p.to(device) for p in self.collected_params]

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into the given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return {'decay': self.decay, 'num_updates': self.num_updates, 'shadow_params': self.shadow_params}

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]


class EMACallback(pl.Callback):
    def __init__(self, parameters, dirpath, decay=0.999, freeze_epoch=-1, every_n_train_steps=100):
        super().__init__()
        self.ema = ExponentialMovingAverage(parameters, decay, use_num_updates=True)
        self.freeze_epoch = freeze_epoch
        self.dirpath = dirpath
        self.every_n_train_steps = every_n_train_steps

    def on_train_start(self, trainer, pl_module):
        # Move EMA parameters to the correct device
        self.ema.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch > self.freeze_epoch:
            self.ema.update(pl_module.parameters())
        if trainer.global_step > 0 and trainer.global_step % self.every_n_train_steps == 0:
            ckpt = {"state_dict": self.ema.state_dict(), "global_step": trainer.global_step, "batch_idx": batch_idx}
        if trainer.global_step > 0 and trainer.global_step % self.every_n_train_steps == 0:
            ckpt = {"state_dict": self.ema.state_dict(), "global_step": trainer.global_step, "batch_idx": batch_idx}
            torch.save(ckpt, os.path.join(self.dirpath, f"ema_parameters_epoch_{trainer.current_epoch}.pt"))

    def on_train_epoch_end(self, trainer, pl_module):
        ckpt = {"state_dict": self.ema.state_dict(), "global_step": trainer.global_step}
        torch.save(ckpt, os.path.join(self.dirpath, f"ema_parameters_epoch_{trainer.current_epoch}.pt"))

    def load(self, state_dict, device):
        for param in state_dict['shadow_params']:
            param.requires_grad = True
        self.ema.load_state_dict(state_dict, device)
