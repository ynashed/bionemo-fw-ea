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
import re
from pathlib import Path

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf

from bionemo.model.molecule.moco.data.molecule_datamodule import MoleculeDataModule
from bionemo.model.molecule.moco.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback
from bionemo.model.molecule.moco.models.module import Graph3DInterpolantModel
from bionemo.model.molecule.moco.models.utils_train import EMACallback


@hydra_runner(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    This is the main function conducting data loading and model training.
    """
    logging.info("\n\n************** Experiment Configuration ***********")
    pl.seed_everything(cfg.train.seed)
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    cfg.outdir = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, 'checkpoints'), exist_ok=True)
    if cfg.resume:
        if os.path.isdir(cfg.resume):
            cfg.resume = f"{cfg.resume}/last.ckpt"
            ckpt = "last"
        else:
            ckpt = cfg.resume
        ema_dir = os.path.dirname(cfg.resume)
        if not cfg.ema_resume:
            latest_epoch = max(
                [
                    int(re.search(r"ema_parameters_epoch_(\d+)\.pt", f).group(1))
                    for f in os.listdir(ema_dir)
                    if re.search(r"ema_parameters_epoch_(\d+)\.pt", f)
                ],
                default=-1,
            )
            cfg.ema_resume = f"{ema_dir}/ema_parameters_epoch_{latest_epoch}.pt"
        pl_module = Graph3DInterpolantModel.load_from_checkpoint(cfg.resume)
        ema_state_dict = torch.load(cfg.ema_resume)[
            'state_dict'
        ]  #! Pytorch Saving does not save the device and requires_grad
        ema_callback = EMACallback(
            pl_module.parameters(),
            dirpath=os.path.join(cfg.outdir, 'checkpoints'),
        )
        ema_callback.load(ema_state_dict, pl_module.device)
    else:
        pl_module = Graph3DInterpolantModel(
            loss_params=cfg.loss,
            optimizer_params=cfg.optimizer,
            lr_scheduler_params=cfg.lr_scheduler,
            dynamics_params=cfg.dynamics,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            self_cond_params=OmegaConf.select(cfg, "self_conditioning", default=None),
        )
        ema_callback = EMACallback(pl_module.parameters(), dirpath=os.path.join(cfg.outdir, 'checkpoints'))
        ckpt = None

    logger = pl.loggers.WandbLogger(
        save_dir=cfg.outdir,
        project=cfg.wandb_params.project,
        group=cfg.wandb_params.group,
        name=cfg.run_name,
        id=cfg.run_name,
        resume='must' if cfg.resume is not None else False,
        entity=cfg.wandb_params.entity,
        mode=cfg.wandb_params.mode,
    )
    logger.log_hyperparams(cfg)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, 'checkpoints'),
        save_last=True,
        every_n_train_steps=cfg.train.checkpoint_every_n_train_steps,
        save_on_train_epoch_end=True,
        filename="last-{epoch}-{step}",
    )
    metric_name = cfg.train.checkpoint_monitor.replace("/", "_")
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, 'checkpoints'),
        save_top_k=5,
        monitor=cfg.train.checkpoint_monitor,
        mode=cfg.train.checkpoint_monitor_mode,
        filename="best-{epoch}-{step}--{" + metric_name + ":.3f}",
    )
    evaluation_callback = MoleculeEvaluationCallback(
        n_graphs=cfg.evaluation.n_molecules, batch_size=cfg.evaluation.batch_size, timesteps=cfg.evaluation.timesteps
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.n_epochs,
        logger=logger,
        callbacks=[lr_monitor, ema_callback, evaluation_callback, last_checkpoint_callback, best_checkpoint_callback],
        enable_progress_bar=cfg.train.enable_progress_bar,
        accelerator='gpu',
        devices=cfg.train.gpus,
        strategy=('ddp' if cfg.train.gpus > 1 else 'auto'),
        check_val_every_n_epoch=cfg.train.val_freq,
        gradient_clip_val=cfg.train.gradient_clip_value,
        log_every_n_steps=cfg.train.log_freq,  # for train steps
        num_sanity_val_steps=0,  # skip sanity since sampling from random weights causes explosion in discrete elements
    )

    datamodule = MoleculeDataModule(**cfg.data)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
