# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from bionemo.model.molecule.moco.data.molecule_datamodule import MoleculeDataModule
from bionemo.model.molecule.moco.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback
from bionemo.model.molecule.moco.models.module import Graph3DInterpolantModel
from bionemo.model.molecule.moco.models.utils_train import ExponentialMovingAverage as EMA


if __name__ == "__main__":
    # ckpt_path = "/workspace/bionemo/results/jodo/jodo_slurm/checkpoints/last.ckpt"
    # ckpt_path = "/workspace/bionemo/results/eqgatdiff/eqgat_slurm/checkpoints/last.ckpt"
    res_dir = "/workspace/bionemo/results/eqgatdiff/eqgat_big_self_cond/checkpoints"
    ckpt_path = f"{res_dir}/epoch=214-step=315199.ckpt"
    ema_weights = f"{res_dir}/ema_parameters_epoch_214.pt"
    n_graphs = 100
    batch_size = 100
    # to load previous weights
    # from omegaconf import OmegaConf
    # cfg = OmegaConf.load('/workspace/bionemo/examples/molecule/moco/conf/train_eqgat.yaml')
    # model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path, loss_params=cfg.loss)
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
    datamodule = MoleculeDataModule(
        dataset_root="/workspace/bionemo/data/pyg_geom_drug",
        processed_folder="processed",
        batch_size=150,
        inference_batch_size=150,
        removed_h=False,
        data_loader_type="midi",
    )

    eval_callback = MoleculeEvaluationCallback(
        n_graphs=n_graphs,
        batch_size=batch_size,
        timesteps=500,
        compute_train_data_metrics=False,
        train_smiles=datamodule.train_dataset.smiles,
        statistics=datamodule.statistics,
    )

    if ema_weights is not None:
        ema_weights = torch.load(ema_weights, map_location="cuda")
        ema = EMA(model.parameters(), decay=0.999)
        ema.load_state_dict(ema_weights["state_dict"], model.device)
        ema.copy_to(model.parameters())

    model.cuda()
    model.eval()

    eval_callback.evaluate_molecules(model)
