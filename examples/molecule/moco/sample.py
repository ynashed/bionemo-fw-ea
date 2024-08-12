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
    # res_dir = "/workspace/bionemo/results/eqgatdiff/eqgat_big_self_cond/checkpoints"
    # ckpt_path = f"{res_dir}/epoch=214-step=315199.ckpt"
    # ema_weights = f"{res_dir}/ema_parameters_epoch_214.pt"
    ema_weights = None
    ckpt_path = (
        "/workspace/bionemo/examples/molecule/moco/checkpoints/best-epoch=149-step=146708--mol_stable=0.978.ckpt"
    )
    n_graphs = 100
    batch_size = 100
    # to load previous weights
    # from omegaconf import OmegaConf
    # cfg = OmegaConf.load('/workspace/bionemo/examples/molecule/moco/conf/train_eqgat.yaml')
    # model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path, loss_params=cfg.loss)
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path).cuda()
    print(f"loaded model successfully {ckpt_path} on {model.device}")
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Convert the number of parameters to millions
    total_params_in_millions = total_params / 1e6

    # Print the number of parameters
    print(f"Total number of parameters: {total_params_in_millions:.2f}M")
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
        compute_dihedrals=True,
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
    import ipdb

    ipdb.set_trace()
    result = eval_callback.evaluate_molecules(model)
    import ipdb

    ipdb.set_trace()


# TODO: Error needs fixing as model weights require same data path
#   File "/workspace/bionemo/examples/molecule/moco/sample.py", line 33, in <module>
#     model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path)
#   File "/usr/local/lib/python3.10/dist-packages/lightning/pytorch/utilities/model_helpers.py", line 125, in wrapper
#     return self.method(cls, *args, **kwargs)
#   File "/usr/local/lib/python3.10/dist-packages/lightning/pytorch/core/module.py", line 1581, in load_from_checkpoint
#     loaded = _load_from_checkpoint(
#   File "/usr/local/lib/python3.10/dist-packages/lightning/pytorch/core/saving.py", line 91, in _load_from_checkpoint
#     model = _load_state(cls, checkpoint, strict=strict, **kwargs)
#   File "/usr/local/lib/python3.10/dist-packages/lightning/pytorch/core/saving.py", line 158, in _load_state
#     obj = cls(**_cls_kwargs)
#   File "/workspace/bionemo/bionemo/model/molecule/moco/models/module.py", line 158, in __init__
#     self.interpolants = self.initialize_interpolants()
#   File "/workspace/bionemo/bionemo/model/molecule/moco/models/module.py", line 213, in initialize_interpolants
#     interp_param["custom_prior"] = self.load_prior(interp_param["custom_prior"]).float()
#   File "/workspace/bionemo/bionemo/model/molecule/moco/models/module.py", line 196, in load_prior
#     array = np.load(fpath)
#   File "/usr/local/lib/python3.10/dist-packages/numpy/lib/npyio.py", line 405, in load
#     fid = stack.enter_context(open(os_fspath(file), "rb"))
# FileNotFoundError: [Errno 2] No such file or directory: '/data/pyg_geom_drug/processed/train_types_h.npy'
# when loading you can overwrite any hyperparameters (eg, interpollant_params).
# pl_module = Graph3DInterpolantModel.load_from_checkpoint(cfg.resume, interpolant_params=cfg.interpolant)
