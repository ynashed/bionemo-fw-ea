# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pickle

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
    ema_weights = '/workspace/bionemo/examples/molecule/moco/checkpoints/SLOW_BEST_ema_parameters_epoch_239.pt'
    ckpt_path = '/workspace/bionemo/examples/molecule/moco/checkpoints/SLOW_BEST_best-epoch=219-step=215172--mol_stable=0.975.ckpt'  # best-epoch=549-step=403162--mol_stable=0.979.ckpt'  # best-epoch=169-step=166269--mol_stable=0.984.ckpt'  # best-epoch=149-step=146708--mol_stable=0.978.ckpt"

    n_graphs = 5000
    batch_size = 200
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
        batch_size=batch_size,
        inference_batch_size=batch_size,
        removed_h=False,
        data_loader_type="midi",
    )

    # eval_callback = MoleculeEvaluationCallback(
    #     n_graphs=n_graphs,
    #     batch_size=batch_size,
    #     timesteps=500,
    #     compute_train_data_metrics=False,
    #     compute_dihedrals=True,
    #     train_smiles=datamodule.train_dataset.smiles,
    #     statistics=datamodule.statistics,
    # )
    ema_weight_path = ema_weights
    if ema_weights is not None:
        ema_weights = torch.load(ema_weights, map_location="cuda")
        ema = EMA(model.parameters(), decay=0.999)
        ema.load_state_dict(ema_weights["state_dict"], model.device)
        ema.copy_to(model.parameters())

    model.cuda()
    model.eval()
    for i in range(2, 8):
        eval_callback = MoleculeEvaluationCallback(
            n_graphs=n_graphs,
            batch_size=batch_size,
            timesteps=500,
            compute_train_data_metrics=False,
            compute_dihedrals=True,
            train_smiles=datamodule.train_dataset.smiles,
            statistics=datamodule.statistics,
        )
        result = eval_callback.evaluate_molecules(model, return_molecules=True)
        result['ckpt'] = ckpt_path
        result['ema_path'] = ema_weight_path
        save_path = '/workspace/bionemo/bionemo/model/molecule/moco/models/results/megalodon/'
        with open(save_path + f"5k_mols_slow_mega{i}.pkl", 'wb') as f:
            pickle.dump(result, f)
    # import ipdb

    # ipdb.set_trace()


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
# mol_stable 0.9714000225067139
# atm_stable 0.9992256164550781
# validity 0.9544000029563904
# stable_valid 0.949
# not_stable_valid 0.0054
# stable_not_valid 0.0224
# not_stable_not_valid 0.0232
# QED 0.6312966802754982
# SA 0.6583109807208718
# LogP 2.5941492120704117
# Lipinski 4.902975691533948
# Diversity 0.8856248777579252
# Unique 1.0
# bond_lengths 0.08495713770389557
# bond_angles 0.506560742855072
# dihedrals 1.2173022031784058
