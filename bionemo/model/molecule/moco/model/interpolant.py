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
from torch_scatter import scatter_mean


class ContinuousInterpolant:
    """
    Class for continuous interpolation.

    Attributes:
        schedule_type (str): Type of interpolant schedule.
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
    """

    def __init__(
        self,
        method_type: str,
        schedule_type: str,
        prior_type: str,
        update_weight_type: str,
        solver_type: str = "ode",
        timesteps: int = 500,
    ):
        """
        Initialize ContinuousFlowMatching instance.

        Args:
            method_type (str): diffusion or flow_matching
            schedule_type (str): Type of interpolant schedule.
            prior_type (str): Type of prior.
            update_weight_type (str): Type of interpolant update weight.
        """
        self.method_type = method_type
        self.schedule_type = schedule_type
        self.prior_type = prior_type
        self.update_weight_type = update_weight_type
        self.init_schedulers(method_type, schedule_type, timesteps)
        self.timesteps = timesteps
        self.solver_type = solver_type
        # TODO do we need forward and reverse typing or can it be the same for both?

    def init_schedulers(self, method_type, schedule_type, tiemsteps):
        self.forward_data_schedule = []
        self.forward_noise_schedule = []
        self.reverse_data_schedule = []
        self.reverse_noise_schedule = []

    def update_weight(self, t):
        if self.update_weight_type == "constant":
            weight = 1
        elif self.update_weight_type == "recip_time_to_go":
            weight = 1 / (1 - t)
        return weight

    def forward_schedule(self, t):
        if self.method_type == 'diffusion':
            t = 1 - t
        return self.forward_data_schedule[t], self.forward_noise_schedule[t]

    def reverse_schedule(self, t):
        if self.method_type == 'diffusion':
            t = 1 - t
            return self.reverse_data_schedule[t], self.reverse_noise_schedule[t].self.log_var[t]
        elif self.method_type == 'flow_matching':
            return self.update_weight(t)

    def interpolate(self, x1, t, batch, com_free=True):
        """
        Interpolate using continuous flow matching method.
        """
        data_scale, noise_scale = self.forward_schedule(t)
        return x1, data_scale * x1 + noise_scale * self.prior(x1.shape, batch, com_free).to(x1.device)

    def prior(self, shape, batch, com_free):
        if self.prior_type == "gaussian" or self.prior_type == "normal":
            x0 = torch.randn(shape)
            if com_free:
                x0 = x0 - scatter_mean(x0, batch, dim=0)[batch]
        else:
            raise ValueError("Only Gaussian is supported")
        return x0

    def step(self, xt, x_hat, t, batch, dt=None, s=None):
        """
        Perform a euler step in the continuous flow matching method.
        """
        if self.method_type == "diffusion":
            if self.solver_type == "sde":
                data_scale, noise_scale, log_var = self.reverse_schedule(t)
                # data_scale = extract(self.posterior_mean_c0_coef, t, batch)
                # noise_scale = extract(self.posterior_mean_ct_coef, t, batch)
                # pos_log_variance = extract(self.posterior_logvar, t, batch)
                mean = data_scale * x_hat + noise_scale * xt
                # no noise when diffusion t == 0 so flow matching t == 1
                nonzero_mask = (t == 1).float()[batch].unsqueeze(-1)
                # ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                x_next = mean + nonzero_mask * (0.5 * log_var).exp() * torch.randn_like(xt)
                # TODO: can add guidance module here
        elif self.method_type == "flow_matching":
            if self.schedule_type == "linear":
                x_next = xt + self.reverse_schedule(t) * dt * (x_hat - xt)
            # TODO: write this in a way that also can accept the scheduler of FlowMol
            # ! This wil require refactoring to t_cur and t_next and defining it this way
        return x_next
