# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch_scatter import scatter_mean

from bionemo.model.molecule.moco.interpolant.interpolant import Interpolant
from bionemo.model.molecule.moco.interpolant.interpolant_scheduler import build_scheduler
from bionemo.model.molecule.moco.interpolant.ot import align_structures, pairwise_distances, permute_and_slice


class ContinuousFlowMatchingInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        prior_type (str): Type of prior.
        vector_field_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        prior_type: str = 'gaussian',
        vector_field_type: str = "standard",
        solver_type: str = "ode",
        timesteps: int = 500,
        min_t: float = 1e-2,
        time_type: str = 'continuous',
        num_classes: int = 3,
        scheduler_type='linear',
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
        com_free: bool = True,
        noise_sigma: float = 0.0,
        optimal_transport: str = None,
        clip_t: float = 0.9,
        loss_weight_type: str = 'standard',  # 'uniform'
        loss_t_scale: float = 0.1,  # this makes max scale 1
    ):
        super(ContinuousFlowMatchingInterpolant, self).__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.vector_field_type = vector_field_type
        self.min_t = min_t
        self.com_free = com_free
        self.noise_sigma = noise_sigma
        self.optimal_transport = optimal_transport
        self.init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip)
        self.max_t = 1.0 - min_t
        self.clip_t = clip_t
        self.loss_weight_type = loss_weight_type
        self.loss_t_scale = loss_t_scale

    def init_schedulers(self, timesteps, scheduler_type, s, sqrt, nu, clip):
        self.schedule_type = scheduler_type
        if scheduler_type == "linear":  #! vpe_linear is just linear with an update weight of recip_time_to_go
            self.discrete_time_only = False
            time = torch.linspace(self.min_t, 1, self.timesteps)
            self.register_buffer("time", time)
            self.register_buffer("forward_data_schedule", time)
            self.register_buffer("forward_noise_schedule", 1.0 - time)
        elif scheduler_type == "vpe":
            # ! Doing this enforces discrete_time_only
            self.discrete_time_only = True
            # self.alphas, self.alphas_prime = cosine_beta_schedule_fm(
            #     schedule_params, timesteps
            # )  # FlowMol defines alpha as 1 - cos ^2
            self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip)
            alphas, betas = self.scheduler.get_alphas_and_betas()
            self.register_buffer('alphas', alphas)
            self.register_buffer('betas', betas)
            self.register_buffer('alpha_bar', alphas)
            self.register_buffer('forward_data_schedule', alphas)
            self.register_buffer('reverse_data_schedule', 1.0 - self.alphas)

    def loss_weight_t(self, time):
        if self.loss_weight_type == "uniform":
            return torch.ones_like(time).to(time.device)

        if self.time_type == "continuous":
            # loss scale for "frameflow": # [1, 0.1] for T = [0, 1]
            return (self.loss_t_scale * (1 / (1 - torch.clamp(time, self.min_t, self.clip_t)))) ** 2
        else:
            if self.schedule_type == "linear":
                t = time / self.timesteps
                return (self.loss_t_scale * (1 / (1 - torch.clamp(t, self.min_t, self.clip_t)))) ** 2
            else:
                return torch.clamp(self.snr(time), min=0.05, max=1.5)

    def update_weight(self, t):
        if self.vector_field_type == "endpoint":
            weight = torch.ones_like(t).to(t.device)
        elif self.vector_field_type == "standard":
            weight = 1 / (1 - torch.clamp(t, self.min_t, self.max_t))
        return weight

    def forward_schedule(self, batch, time):
        if self.time_type == "continuous":
            if self.schedule_type == "linear":
                return time[batch].unsqueeze(1), (1.0 - time)[batch].unsqueeze(1)
            else:
                raise NotImplementedError("Continuous time is only implemented with linear schedule")
        else:
            return (
                self.forward_data_schedule[time].unsqueeze(1)[batch],
                self.forward_noise_schedule[time].unsqueeze(1)[batch],
            )

    def reverse_schedule(self, batch, time, dt):
        if self.time_type == "continuous":
            if self.schedule_type == "linear":
                data_scale = self.update_weight(time[batch]) * dt
        else:
            if self.schedule_type == "linear":
                t = self.forward_data_schedule[time]
                data_scale = self.update_weight(t[batch]) * dt
            elif self.schedule_type == "vpe":  # FlowMol
                data_scale = (
                    self.derivative_forward_data_schedule[time] * dt / (1 - self.forward_data_schedule[time])
                )[
                    batch
                ]  # alpha_prime[t]*dt/(1 - alpha[t]) #! EquiFm uses (1-a)^2 could be due to the definition of the scheduler FloMol uses cosine wheres EquiFm uses exp(- 0.5 * integral of betas(s)) where beta is some noise scheduler funtion

        return data_scale.unsqueeze(1), (1 - data_scale).unsqueeze(1)

    def equivariant_ot_prior(self, batch, data_chunk):
        """Permute the from_mols batch so that it forms an approximate mini-batch OT map with to_mols"""
        #! prior has to be as big as the largets input and then we throw stuff away
        #! noise_batch is a list of beath elements of max atom num x 3
        batch_size = int(max(batch) + 1)
        data_batch = [data_chunk[batch == idx] for idx in range(batch_size)]
        max_num_atoms = max([x.shape[0] for x in data_batch])
        noise_batch = [self.prior(None, (max_num_atoms, 3), data_chunk.device) for _ in range(batch_size)]
        # if scale:
        #      noise_batch = [x*0.2*np.log(max_num_atoms + 1) for x in noise_batch]
        mol_matrix = []
        cost_matrix = []

        # Create matrix with data on outer axis and noise on inner axis
        for data in data_batch:
            best_noise = [permute_and_slice(noise, data) for noise in noise_batch]
            sub_batch = torch.arange(len(noise_batch)).repeat_interleave(data.shape[0])
            best_noise = align_structures(torch.cat(best_noise, dim=1), sub_batch, data, broadcast_reference=True)
            best_noise = best_noise.reshape((len(noise_batch), data.shape[0], 3))
            best_costs = pairwise_distances(
                best_noise, data.repeat(len(noise_batch), 1).reshape(len(noise_batch), data.shape[0], 3)
            )[
                :, 0
            ]  # B x 1
            mol_matrix.append(best_noise)  # B x N x 3
            cost_matrix.append(best_costs.numpy())

        row_indices, col_indices = linear_sum_assignment(np.array(cost_matrix))
        optimal_noise = [mol_matrix[r][c] for r, c in zip(row_indices, col_indices)]
        return torch.cat(optimal_noise, dim=-1)  #! returns N tot x 3 where this matches data_chunk

    def interpolate(self, batch, x1, time):
        """
        Interpolate using continuous flow matching method.
        """
        if self.optimal_transport in ["equivariant_ot", "scale_ot"]:
            x0 = self.equivariant_ot_prior(batch, x1)
        else:
            x0 = self.prior(batch, x1.shape, x1.device)
        data_scale, noise_scale = self.forward_schedule(batch, time)
        if self.noise_sigma > 0:
            interp_noise = self.prior(batch, x1.shape, x1.device) * self.noise_sigma
        else:
            interp_noise = 0
        return x1, data_scale * x1 + noise_scale * x0 + interp_noise, x0

    def vector_field(self, batch, x1, xt, time):
        """
        Return (x1 - xt) / (1 - t)
        """
        return (x1 - xt) / (1.0 - torch.clamp(time[batch], self.min_t, self.max_t))

    def prior(self, batch, shape, device, x1=None):
        if self.prior_type == "gaussian" or self.prior_type == "normal":
            x0 = torch.randn(shape).to(device)
            if self.com_free:
                if batch:
                    x0 = x0 - scatter_mean(x0, batch, dim=0)[batch]
                else:
                    x0 = x0 - x0.mean(0)
        else:
            raise ValueError("Only Gaussian is supported")
        if self.optimal_transport == "scale_ot":
            if batch:
                scale = 0.2 * torch.log(torch.bincount(batch) + 1)[batch]
            else:
                scale = 0.2 * torch.log(x0.shape[0] + 1)
            x0 = x0 * scale
        return x0.to(device)

    def step(self, batch, xt, x_hat, time, x0=None, dt=None):
        """
        Perform a euler step in the continuous flow matching method.
        Here we allow two options for the choice of update: vector field as a function of t and end point vector field
         A) VF = x1 - xt /(1-t) --> x_next = xt + 1/(1-t) * dt * (x_hat - xt) see Lipman et al. https://arxiv.org/pdf/2210.02747
         B) Linear with dynamics as data prediction VF = x1 - x0 --> x_next = xt +  dt * (x_hat - x0) see Tong et al. https://arxiv.org/pdf/2302.00482 sec 3.2.2 basic I-CFM
        Both of which can add additional noise.
        """
        if self.vector_field_type == "standard":
            data_scale, noise_scale = self.reverse_schedule(
                batch, time, dt
            )  #! this is same as xt + vf*df where vf = (xhat-xt)/(1-t) and can use the vector_field function
            x_next = data_scale * x_hat + noise_scale * xt
        elif self.vector_field_type == "endpoint":
            data_scale, _ = self.reverse_schedule(batch, time, dt)
            x_next = xt + data_scale * (x_hat - x0)
        else:
            raise ValueError(f"f{self.vector_field_type} is not a recognized vector_field_type")

        if self.noise_sigma > 0:
            x_next += self.prior(batch, x_hat.shape, x_hat.device) * self.noise_sigma  # torch.randn_like(x_hat)
        return x_next
