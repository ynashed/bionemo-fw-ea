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
    run_name = "ord_megalodon_bs150_clip_dotfn"
    res_dir = f"/results/{run_name}/checkpoints"
    weight_name = "best-epoch=219-step=215172--mol_stable=0.975.ckpt"
    ckpt_path = f"{res_dir}/{weight_name}"
    ema_weights = None  # f"{res_dir}/ema_parameters_epoch_214.pt"
    n_graphs = 5000
    batch_size = 100
    import ipdb

    ipdb.set_trace()
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
    datamodule = MoleculeDataModule(
        dataset_root="/data/pyg_geom_drug",
        processed_folder="processed",
        batch_size=batch_size,
        inference_batch_size=batch_size,
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

    results = eval_callback.evaluate_molecules(model)
    import ipdb

    ipdb.set_trace()
    print(results)
