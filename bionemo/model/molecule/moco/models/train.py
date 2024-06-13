# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from lightning import pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf

from bionemo.model.molecule.moco.data.molecule_datamodule import MoleculeDataModule
from bionemo.model.molecule.moco.models.module import Graph3DInterpolantModel


@hydra_runner(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    This is the main function conducting data loading and model training.
    """
    logging.info("\n\n************** Experiment Configuration ***********")
    pl.seed_everything(cfg.train.seed)
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    if cfg.resume:
        pl_module = Graph3DInterpolantModel.load_from_checkpoint(cfg.resume)
    else:
        pl_module = Graph3DInterpolantModel(
            loss_params=cfg.loss,
            optimizer_params=cfg.optimizer,
            lr_scheduler_params=cfg.lr_scheduler,
            dynamics_params=cfg.dynamics,
            interpolant_params=cfg.interpolant,
        )

    # logger = pl.loggers.WandbLogger(
    #     save_dir=cfg.outdir,
    #     project=cfg.wandb_params.project,
    #     group=cfg.wandb_params.group,
    #     name=cfg.run_name,
    #     id=wandb.util.generate_id(),
    #     resume='must' if cfg.resume is not None else False,
    #     entity=cfg.wandb_params.entity,
    #     mode=cfg.wandb_params.mode,
    # )
    # logger.log_hyperparams(cfg)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=Path(out_dir, 'checkpoints'),
    #     filename="best-model-{epoch:04d}",
    #     monitor="val/loss_epoch",
    #     save_top_k=1,
    #     save_last=True,
    #     mode="min",
    # )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpointing = ModelCheckpoint(every_n_epochs=1, monitor="train-loss", mode="min", save_last=True)

    trainer = pl.Trainer(
        max_epochs=cfg.train.n_epochs,
        # logger=logger,
        callbacks=[lr_monitor, checkpointing],
        enable_progress_bar=cfg.train.enable_progress_bar,
        accelerator='gpu',
        devices=1,  # cfg.train.gpus,
        strategy=('ddp' if cfg.train.gpus > 1 else 'auto'),
        check_val_every_n_epoch=1,  # cfg.train.val_freq,
        gradient_clip_val=cfg.train.gradient_clip_value,
    )

    datamodule = MoleculeDataModule(cfg.data)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
