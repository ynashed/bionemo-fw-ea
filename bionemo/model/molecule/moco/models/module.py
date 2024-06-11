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
from lightning import pytorch as pl
from nemo.core.config import hydra_runner
from omegaconf import DictConfig
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from torch_sparse import coalesce

from bionemo.model.molecule.moco.data.molecule_datamodule import MoleculeDataModule
from bionemo.model.molecule.moco.models.interpolant import build_interpolant
from bionemo.model.molecule.moco.models.moco import MoCo

# from bionemo.model.molecule.moco.models.mpnn import TimestepEmbedder, AdaLsN
from bionemo.model.molecule.moco.models.utils import InterpolantLossFunction


class Graph3DInterpolantModel(pl.LightningModule):
    def __init__(
        self,
        loss_params: DictConfig,
        optimizer_params: DictConfig,
        dynamics_params: DictConfig,
        interpolant_params: DictConfig,
    ):
        # import ipdb; ipdb.set_trace()
        super(Graph3DInterpolantModel, self).__init__()
        self.optimizer_params = optimizer_params
        self.dynamics_params = dynamics_params
        self.interpolant_params = interpolant_params
        self.loss_params = loss_params.loss_scale
        self.loss_function = InterpolantLossFunction(use_distance=loss_params.use_distance)
        self.interpolants = self.initialize_interpolants()
        self.dynamics = MoCo()

    def setup(self, stage):
        for interpolant in self.interpolants.values():
            interpolant.to(self.device)

    def initialize_interpolants(self):
        interpolants = {}
        for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
            index = interp_param.variable_name
            interpolants[index] = build_interpolant(**interp_param)
        return interpolants

    def configure_optimizers(self):
        if self.optimizer_params.type == "adamw":
            optimizer = torch.optim.AdamW(
                self.dynamics.parameters(),
                lr=self.optimizer_params.lr,
                amsgrad=self.optimizer_params.amsgrad,
                weight_decay=self.optimizer_params.weight_decay,
            )
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizer_params.type)

        return optimizer

    def sample_time(self, batch):
        batch_size = int(batch.batch.max()) + 1
        time = self.interpolants['x'].sample_time(
            num_samples=batch_size,
            device=batch.x.device,
            method=self.interpolant_params.sample_time_method,
            mean=self.interpolant_params.sample_time_mean,
            scale=self.interpolant_params.sample_time_scale,
        )
        return batch_size, time

    def pre_format_molecules(self, batch, batch_size):
        batch['x'] = batch['x'] - scatter_mean(batch['x'], index=batch.batch, dim=0, dim_size=batch_size)[batch.batch]

        if self.interpolants['h'].prior_type in ["mask", "absorb"]:
            batch["h"] = torch.cat((batch["h"], torch.zeros((batch["h"].size(0), 1)).to(batch["h"].device)), dim=1).to(
                batch["h"].device
            )
        # Load bond information from the dataloader
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=batch.edge_index, edge_attr=batch.edge_attr, sort_by_row=False
        )
        # Create Fully Connected Graph instead
        edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        edge_attr_tmp = torch.full(
            size=(edge_index_global.size(-1),),
            fill_value=0,
            device=edge_index_global.device,
            dtype=torch.long,
        )
        edge_index_global = torch.cat([edge_index_global, bond_edge_index], dim=-1)
        edge_attr_tmp = torch.cat([edge_attr_tmp, bond_edge_attr], dim=0)
        edge_index_global, edge_attr_global = coalesce(
            index=edge_index_global, value=edge_attr_tmp, m=batch['x'].size(0), n=batch['x'].size(0), op="max"
        )
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
        )
        if self.interpolants['h'].prior_type in ["mask", "absorb"]:
            batch["h"] = self.add_absorbption_state(batch["h"])

        if self.interpolants['edge_attr'].prior_type in ["mask", "absorb"]:
            # TODO: should we use the no bond state as the mask? or create an extra dim
            edge_attr_global = self.add_absorbption_state(edge_attr_global)

        batch['edge_attr'] = edge_attr_global
        batch['edge_index'] = edge_index_global

        # TODO: anymore specfic shifting of molecule only data
        return batch

    def add_adsorbtion_state(self, h):
        # h is (N, C) and we want to add a column of all zeros at the end
        N, C = h.shape
        zeros_column = torch.zeros(N, 1, device=h.device)
        return torch.cat([h, zeros_column], dim=1)

    def interpolate(self, batch, time):
        for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
            interpolant = self.interpolants[f"{interp_param.variable_name}"]
            if interpolant is None:
                batch[f"{interp_param.variable_name}_t"] = batch[f"{interp_param.variable_name}"]
            if interp_param.variable_name == "edge_attr":
                if self.interpolant_params.time_type == "discrete":
                    _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate_edges(
                        batch[f"{interp_param.variable_name}"], batch["edge_index"], batch.batch, t_idx=time
                    )
                elif self.interpolant_params.time_type == "continuous":
                    _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate_edges(
                        batch[f"{interp_param.variable_name}"], batch["edge_index"], batch.batch, t=time
                    )
            else:
                if self.interpolant_params.time_type == "discrete":
                    _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate(
                        batch[f"{interp_param.variable_name}"], batch.batch, t_idx=time
                    )
                elif self.interpolant_params.time_type == "continuous":
                    _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate(
                        batch[f"{interp_param.variable_name}"], batch.batch, t=time
                    )
        return batch

    # def inject_time(self, batch, batch_time):
    #     #! For now tis function only puts the time into H
    #     for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
    #         if interp_param.variable_name.lower() == "h":
    #             time_embedding = self.time_embedding(batch_time[interpolant_idx][batch.batch])
    #             batch[f"{interp_param.variable_name}_t"] = self.adaptive_layer_norm(batch[f"{interp_param.variable_name}_t"], time_embedding)
    #     return batch

    def aggregate_discrete_variables(self, results):
        # TODO how do we handle the charge and other H add on logic
        return results

    def separate_discrete_variables(self, results):
        # TODO how do we handle the charge and other H add on logic
        return results

    def training_step(self, batch, batch_idx):
        out, time = self(batch)
        batch_geo = batch.batch
        # TODO turn this in to an iterative loop where if the interpolant is non None we take a loss
        ws_t = self.interpolants['x'].snr_loss_weight(time)
        # loss = 0
        # for var_name, interpolant in self.interpolants.items():
        #     if 'edge' in var_name:
        #         loss_component, edge_attr_out = self.loss_function.edge_loss(batch_geo, out['edge_attr_hat'], out['edge_attr'],index = batch['edge_index'][1], num_atoms=x_pred.size(0), batch_weight = ws_t, scale = self.loss_scalar['edge_attr'])
        #     else:
        #         loss_component, pred = self.loss_function(batch_geo, out[f'{var_name}_hat'], out[f'{var_name}'], batch_weight = ws_t, scale = self.loss_scalar[f'{var_name}'])
        #     loss += loss_component
        x_loss, x_pred = self.loss_function(
            batch_geo, out['x_hat'], out['x'], batch_weight=ws_t, scale=self.loss_scalar['x']
        )
        h_loss, h_out = self.loss_function(
            batch_geo, out['h_hat'], out['h'], continuous=False, batch_weight=ws_t, scale=self.loss_scalar['h']
        )
        edge_loss, edge_attr_out = self.loss_function.edge_loss(
            batch_geo,
            out['edge_attr_hat'],
            out['edge_attr'],
            index=batch['edge_index'][1],
            num_atoms=x_pred.size(0),
            batch_weight=ws_t,
            scale=self.loss_scalar['edge_attr'],
        )
        loss = x_loss + h_loss + edge_loss
        if self.loss_function.use_distance:
            if "Z_hat" in out.keys() and self.loss_function.use_distance == "triple":
                z_hat = out["Z_hat"]
            else:
                z_hat = None
            distance_loss_tp, distance_loss_tz, distance_loss_pz = self.loss_function.distance_loss(
                batch_geo, batch['x'], x_pred, z_hat
            )
            distance_loss = distance_loss_tp + distance_loss_tz + distance_loss_pz
            loss = loss + distance_loss
        return loss

    def forward(self, batch):
        """
        This forward function assumes we are doing some form (including none) of interpolation on positions X, node features H, and edge attributes edge_attr.
        1. Sample time from the distribution that is defined via the X interpolant params
        2. Shift X to 0 CoM, add absorbing state for H, create fully connected graph and edge features for edge_attr
                - TODO: Also want to do any charge/other feature preprocessing eventaullly
        3. Interpolate all needed variables which are defined by "string" args in the config.
        4. Aggregate all the discrete non edge features in the H variable for modeling.
        5. Dynamics forward pass to predict clean data given noisy data.
        6. Seperate the aggregated discrete predictions for easier loss calculation.
        """
        batch_size, time = self.sample_time(batch)
        batch = self.pre_format_molecules(batch, batch_size)
        batch = self.interpolate(batch, time)
        #    batch = self.initialize_pair_embedding(batch) #! Do in the dynamics
        batch = self.aggregate_discrete_variables(batch)
        import ipdb

        ipdb.set_trace()
        out = self.dynamics(
            batch=batch.batch,
            X=batch["x_t"],
            H=batch["h_t"],
            E=batch["edge_attr_t"],
            E_idx=batch["edge_index"],
            t=time,
        )
        out = self.separate_discrete_variables(out)
        return out, time


@hydra_runner(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    # import ipdb; ipdb.set_trace()
    datamodule = MoleculeDataModule(cfg.data)
    train_dataloader = datamodule.test_dataloader()
    device = 'cuda:0'
    model = Graph3DInterpolantModel(cfg.loss, cfg.optimizer, cfg.dynamics, cfg.interpolants).to(device)
    model.setup("fit")  #! this has to be called after model is moved to device for everything to propagate
    import ipdb

    ipdb.set_trace()
    for batch in train_dataloader:
        batch.h = batch.x
        batch.x = batch.pos
        batch.pos = None

        batch = batch.to(device)
        print(batch)
        out, time = model(batch)
        # batch = batch.batch,
        #                   X = batch.x_t,
        #                   H = batch.h_t,
        #                   E = batch.edge_attr_t,
        #                   E_idx = batch.edge_index,
        #                   t = time
        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    main()
