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
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch_geometric.utils import sort_edge_index
from torch_scatter import scatter_mean

from bionemo.model.molecule.moco.models.interpolant_scheduler import build_scheduler
from bionemo.model.molecule.moco.models.interpolant_utils import (
    float_time_to_index,
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_sample_categorical,
)
from bionemo.model.molecule.moco.models.ot import align_structures, pairwise_distances, permute_and_slice


def build_interpolant(
    interpolant_type: str,
    prior_type: str = "uniform",
    update_weight_type: str = "constant",
    diffusion_type: str = "d3pm",
    solver_type: str = "sde",
    time_type: str = 'continuous',
    scheduler_type='cosine_adaptive',
    scheduler_cut: bool = False,
    s: float = 0.008,
    sqrt: bool = False,
    nu: float = 1.0,
    clip: bool = True,
    timesteps: int = 500,
    num_classes: int = 10,
    min_t: float = 1e-2,
    custom_prior: torch.Tensor = None,
    com_free: bool = True,
    variable_name: str = None,
    concat: str = None,
    offset: int = 0,
    noise_sigma: float = 0.0,
    optimal_transport: str = None
    # TODO: here is where we add all the possible things that could go into any interpolant class
):
    """
     Builds an interpolant for the specified configuration.

    The interpolant is constructed based on various parameters that define the type of interpolation,
    prior distribution, update mechanism, diffusion process, solver method, and other configurations.

    Parameters:
    -----------
    interpolant_type : str
        The type of interpolant to build.
    prior_type : str, optional
        The type of prior distribution. Default is "uniform".
    update_weight_type : str, optional
        The type of update weight to use. Default is "constant".
    diffusion_type : str, optional
        The type of diffusion process. Default is "d3pm".
    solver_type : str, optional
        The type of solver to use. Default is "sde".
    time_type : str, optional
        The type of time representation. Default is 'continuous'.
    scheduler_type : str, optional
        The type of scheduler to use. Default is 'cosine_adaptive'.
    scheduler_cut : bool, optional
        Whether to apply a scheduler cut. Default is False.
    s : float, optional
        A parameter for the scheduler. Default is 0.008.
    sqrt : bool, optional
        Whether to apply a square root transformation. Default is False.
    nu : float, optional
        A parameter for the scheduler. Default is 1.0.
    clip : bool, optional
        Whether to clip the values. Default is True.
    timesteps : int, optional
        The number of timesteps. Default is 500.
    num_classes : int, optional
        The number of classes. Default is 10.
    min_t : float, optional
        The minimum time value. Default is 1e-2.
    custom_prior : torch.Tensor, optional
        A custom prior distribution. Default is None.
    com_free : bool, optional
        Whether to use a center-of-mass-free configuration. Default is True.
    variable_name : str, optional
        The name of the variable to use. Default is None.
    concat : str, optional
        Concatenation target variable. Default is None.

    Returns:
    --------
    Interpolant
        The constructed interpolant object.

    Notes:
    ------
    The setup for uniform and absorbing priors is assumed to be the same, and the +1 mask state is controlled
    to ensure that the number of classes remains constant in the configuration, representing the desired number
    of classes to model.
    """
    if interpolant_type == "continuous_diffusion":
        return ContinuousDiffusionInterpolant(
            prior_type,
            diffusion_type,
            solver_type,
            timesteps,
            time_type,
            num_classes,
            scheduler_type,
            s,
            sqrt,
            nu,
            clip,
            com_free,
            scheduler_cut,
        )
    elif interpolant_type == "continuous_flow_matching":
        return ContinuousFlowMatchingInterpolant(
            prior_type,
            update_weight_type,
            "ode",
            timesteps,
            min_t,
            time_type,
            num_classes,
            scheduler_type,
            s,
            sqrt,
            nu,
            clip,
            com_free,
            noise_sigma,
            optimal_transport,
        )
    elif interpolant_type == "discrete_diffusion":
        if prior_type in ["absorb", "mask"]:
            num_classes = num_classes + 1
        return DiscreteDiffusionInterpolant(
            prior_type,
            diffusion_type,
            solver_type,
            timesteps,
            time_type,
            num_classes,
            custom_prior,
            scheduler_type,
            s,
            sqrt,
            nu,
            clip,
            scheduler_cut,
        )
    elif interpolant_type == "discrete_flow_matching":
        if prior_type in ["absorb", "mask"]:
            num_classes = num_classes + 1
        return DiscreteFlowMatchingInterpolant(
            prior_type,
            update_weight_type,
            "ode",
            timesteps,
            min_t,
            time_type,
            num_classes,
            custom_prior,
            scheduler_type,
            s,
            sqrt,
            nu,
            clip,
        )
    elif interpolant_type is None:
        #! This is to allow variables we just want to pass in and not noise/denoise
        return None
    else:
        raise NotImplementedError('Interpolant not supported: %s' % interpolant_type)


class Interpolant(nn.Module):
    def __init__(
        self,
        prior_type: str,
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = 'discrete',
    ):
        super(Interpolant, self).__init__()
        self.prior_type = prior_type
        self.timesteps = timesteps
        self.solver_type = solver_type
        self.time_type = time_type

    def sample_time(self, num_samples, method='uniform', device='cpu', mean=0, scale=0.81, min_t=0):
        if self.time_type == "continuous":
            return self.sample_time_continuous(num_samples, method, device, mean, scale, min_t)
        else:
            return self.sample_time_idx(num_samples, method, device, mean, scale)

    def sample_time_idx(self, num_samples, method, device='cpu', mean=0, scale=0.81):
        if method == 'symmetric':
            time_step = torch.randint(0, self.timesteps, size=(num_samples // 2 + 1,))
            time_step = torch.cat([time_step, self.timesteps - time_step - 1], dim=0)[:num_samples]

        elif method == 'uniform':
            time_step = torch.randint(0, self.timesteps, size=(num_samples,))

        elif method == "stab_mode":  #! converts uniform to Stability AI mode distribution

            def fmode(u: torch.Tensor, s: float) -> torch.Tensor:
                return 1 - u - s * (torch.cos((torch.pi / 2) * u) ** 2 - 1 + u)

            time_step = float_time_to_index(fmode(torch.rand(num_samples), scale), self.timesteps)
        elif (
            method == 'logit_normal'
        ):  # see Figure 11 https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf
            time_step = float_time_to_index(
                torch.sigmoid(torch.normal(mean=mean, std=scale, size=(num_samples,))), self.timesteps
            )
        elif method == "beta":
            dist = torch.distributions.Beta(2.0, 1.0)
            time_step = float_time_to_index(dist.sample([num_samples]), self.timesteps)
        else:
            raise ValueError
        return time_step.to(device)

    def sample_time_continuous(self, num_samples, method, device='cpu', mean=0, scale=0.81, min_t=0):
        if method == 'symmetric':
            time_step = torch.rand(num_samples // 2 + 1)
            time_step = torch.cat([time_step, 1 - time_step], dim=0)[:num_samples]

        elif method == 'uniform':
            time_step = torch.rand(num_samples)
        elif method == "stab_mode":  #! converts uniform to Stability AI mode distribution

            def fmode(u: torch.Tensor, s: float) -> torch.Tensor:
                return 1 - u - s * (torch.cos((torch.pi / 2) * u) ** 2 - 1 + u)

            time_step = fmode(torch.rand(num_samples), scale)
        elif (
            method == 'logit_normal'
        ):  # see Figure 11 https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf
            time_step = torch.sigmoid(torch.normal(mean=mean, std=scale, size=(num_samples,)))
        elif method == "beta":  # From MolFlow
            dist = torch.distributions.Beta(2.0, 1.0)
            time_step = dist.sample([num_samples])
        else:
            raise ValueError
        if min_t > 0:
            time_step = time_step * (1 - 2 * min_t) + min_t
        return time_step.to(device)

    def snr_loss_weight(self, time):
        return torch.clamp(self.snr(time), min=0.05, max=1.5)


class ContinuousDiffusionInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        prior_type: str = 'gaussian',
        diffusion_type: str = 'vdm',
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = 'discrete',
        num_classes: int = 3,
        scheduler_type='cosine_adaptive',
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
        com_free: bool = True,
        cut: bool = False,
    ):
        super(ContinuousDiffusionInterpolant, self).__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.diffusion_type = diffusion_type
        self.com_free = com_free
        self.init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip, cut)

    def init_schedulers(self, timesteps, scheduler_type, s, sqrt, nu, clip, cut):
        self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip, cut)
        alphas, betas = self.scheduler.get_alphas_and_betas()
        if self.diffusion_type == "ddpm":
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            alpha_bar = alphas_cumprod = torch.exp(log_alpha_bar)
            alpha_bar_prev = alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            posterior_mean_c0_coef = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            posterior_mean_ct_coef = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
            posterior_logvar = torch.log(
                torch.nn.functional.pad(posterior_variance[:-1], (1, 0), value=posterior_variance[0])
            )

            self.register_buffer('alphas', alphas)
            self.register_buffer('betas', betas)
            self.register_buffer('alpha_bar', alpha_bar)
            self.register_buffer('alpha_bar_prev', alpha_bar_prev)
            self.register_buffer('posterior_variance', posterior_variance)
            self.register_buffer('posterior_mean_c0_coef', posterior_mean_c0_coef)
            self.register_buffer('posterior_mean_ct_coef', posterior_mean_ct_coef)
            self.register_buffer('posterior_logvar', posterior_logvar)

            self.register_buffer('forward_data_schedule', torch.sqrt(alpha_bar))
            self.register_buffer('forward_noise_schedule', torch.sqrt(1 - alpha_bar))
            self.register_buffer('reverse_data_schedule', posterior_mean_c0_coef)
            self.register_buffer('reverse_noise_schedule', posterior_mean_ct_coef)
            self.register_buffer('log_var', posterior_logvar)
            # import pickle
            # temp = {
            #     'forward_data_schedule':  torch.sqrt(alpha_bar),
            #     'forward_noise_schedule': torch.sqrt(1 - alpha_bar),
            #     'reverse_data_schedule': posterior_mean_c0_coef,
            #     'reverse_noise_schedule': posterior_mean_ct_coef,
            #     'log_var': posterior_logvar
            # }
            # with open("/workspace/bionemo/bionemo/model/molecule/moco/models/ddpm.pickle", "wb") as f:
            #     pickle.dump(temp, f)

        elif self.diffusion_type == "vdm":
            assert not cut  # alphas.shape = T + 1
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            alphas_cumprod = torch.exp(log_alpha_bar)
            self.register_buffer('alphas', alphas[1:])
            self.register_buffer('betas', betas[1:])
            # self._log_alpha_bar = log_alpha_bar
            alpha_bar = torch.exp(log_alpha_bar)
            self.register_buffer('alpha_bar', torch.exp(log_alpha_bar)[1:])
            sigma2_bar = -torch.expm1(2 * log_alpha_bar)
            sigma_bar = torch.sqrt(sigma2_bar)
            self.register_buffer('sigma_bar', sigma_bar[1:])

            self.register_buffer('forward_data_schedule', alpha_bar[1:])
            self.register_buffer('forward_noise_schedule', sigma_bar[1:])
            # Diffusion s < t to t is closer to noise T = 0 data
            s_time = list(range(self.timesteps))  # [i for i in range(self.timesteps - 1, -1, -1)]
            t_time = list(range(1, 1 + self.timesteps))

            # sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s, t_int=t)
            s2_s = -torch.expm1(2 * log_alpha_bar[s_time])
            s2_t = -torch.expm1(2 * log_alpha_bar[t_time])
            sigma_sq_ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t)).float()
            self.register_buffer('sigma_sq_ratio', sigma_sq_ratio)

            prefactor1 = sigma2_bar[t_time]
            alpha_pos_ts_sq = torch.exp(2 * log_alpha_bar[t_time] - 2 * log_alpha_bar[s_time])
            prefactor2 = sigma2_bar[s_time] * alpha_pos_ts_sq
            sigma2_t_s = prefactor1 - prefactor2
            noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
            noise_prefactor = torch.sqrt(noise_prefactor_sq)  # .unsqueeze(-1)

            # z_t_prefactor = alpha_pos_ts_sq * sigma_sq_ratio  # .unsqueeze(-1)
            z_t_prefactor = torch.exp(log_alpha_bar[t_time] - log_alpha_bar[s_time]).float() * sigma_sq_ratio
            a_s = alpha_bar[s_time]  # self.get_alpha_bar(t_int=s_int)
            alpha_ratio_sq = alpha_pos_ts_sq  # self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
            sigma_ratio_sq = sigma_sq_ratio  # self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
            x_prefactor = (a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)).float()

            self.register_buffer('reverse_data_schedule', x_prefactor)
            self.register_buffer('reverse_noise_schedule', z_t_prefactor)
            # self.register_buffer('std', noise_prefactor)
            self.register_buffer('log_var', 2 * torch.log(noise_prefactor))  # we do this since its exp(0.5*log_var)
            # import pickle
            # temp = {
            #     'forward_data_schedule': alpha_bar[1:],
            #     'forward_noise_schedule': sigma_bar[1:],
            #     'reverse_data_schedule': x_prefactor,
            #     'reverse_noise_schedule': z_t_prefactor,
            #     'log_var': noise_prefactor
            # }
            # with open("/workspace/bionemo/bionemo/model/molecule/moco/models/vdm.pickle", "wb") as f:
            #     pickle.dump(temp, f)

    def forward_schedule(self, batch, time):
        t_idx = self.timesteps - 1 - time
        return (
            self.forward_data_schedule[t_idx].unsqueeze(1)[batch],
            self.forward_noise_schedule[t_idx].unsqueeze(1)[batch],
        )

    def reverse_schedule(self, batch, time):
        t_idx = self.timesteps - 1 - time
        return (
            self.reverse_data_schedule[t_idx].unsqueeze(1)[batch],
            self.reverse_noise_schedule[t_idx].unsqueeze(1)[batch],
            self.log_var[t_idx].unsqueeze(1)[batch],
        )

    def interpolate(self, batch, x1, time):
        """
        Interpolate using continuous diffusion.
        """
        x0 = self.prior(batch, x1.shape, x1.device)
        data_scale, noise_scale = self.forward_schedule(batch, time)
        return x1, data_scale * x1 + noise_scale * x0, x0

    def prior(self, batch, shape, device, x1=None):
        if self.prior_type == "gaussian" or self.prior_type == "normal":
            x0 = torch.randn(shape).to(device)
            if self.com_free:
                x0 = x0 - scatter_mean(x0, batch, dim=0)[batch]
        else:
            raise ValueError("Only Gaussian is supported")
        return x0.to(device)

    def step(self, batch, xt, x_hat, x0, time, dt=None):
        """
        Perform a euler step.
        """
        if self.solver_type == "sde":
            data_scale, noise_scale, log_var = self.reverse_schedule(batch, time)
            mean = data_scale * x_hat + noise_scale * xt
            if self.diffusion_type == "ddpm":
                # no noise when diffusion t == 0 so flow matching t == 1
                nonzero_mask = (1 - (time == (self.timesteps - 1)).float())[batch].unsqueeze(-1)
                x_next = mean + nonzero_mask * (0.5 * log_var).exp() * self.prior(batch, xt.shape, device=xt.device)
            elif self.diffusion_type == "vdm":
                x_next = mean + (0.5 * log_var).exp() * self.prior(batch, xt.shape, device=xt.device)
            # TODO: can add guidance module here
        else:
            raise ValueError("Only SDE Implemented")

        if self.com_free:
            x_next = x_next - scatter_mean(x_next, batch, dim=0)[batch]
        return x_next

    def snr(self, time):
        t_idx = self.timesteps - 1 - time
        abar = self.alpha_bar[t_idx]
        return abar / (1 - abar)


class ContinuousFlowMatchingInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        prior_type: str = 'gaussian',
        update_weight_type: str = "constant",
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
    ):
        super(ContinuousFlowMatchingInterpolant, self).__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.update_weight_type = update_weight_type
        self.min_t = min_t
        self.com_free = com_free
        self.noise_sigma = noise_sigma
        self.optimal_transport = optimal_transport
        self.init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip)

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
            # self.forward_data_schedule = self.alphas
            self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip)
            alphas, betas = self.scheduler.get_alphas_and_betas()
            self.register_buffer('alphas', alphas)
            self.register_buffer('betas', betas)
            self.register_buffer('alpha_bar', alphas)
            self.register_buffer('forward_data_schedule', alphas)
            self.register_buffer('reverse_data_schedule', 1.0 - self.alphas)

    def snr_loss_weight(self, time):
        if self.time_type == "continuous":
            return torch.clamp(time / (1 - time), min=0.05, max=1.5)
        else:
            if self.schedule_type == "linear":
                t = time / self.timesteps
                return torch.clamp(t / (1 - t), min=0.05, max=1.5)
            else:
                return torch.clamp(self.snr(time), min=0.05, max=1.5)

    def update_weight(self, t):
        if self.update_weight_type == "constant":
            weight = torch.ones_like(t).to(t.device)
        elif self.update_weight_type == "recip_time_to_go":
            weight = torch.clamp(1 / (1 - t), max=self.timesteps)  # at T = 1 this makes data_scale = 1
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

    def step(self, batch, xt, x_hat, time, x0=None, dt=None, last_step=False):
        """
        Perform a euler step in the continuous flow matching method.
        Here we allow two options for the choic of vector field
         A) VF = x1 - xt /(1-t) --> x_next = xt + 1/(1-t) * dt * (x_hat - xt) see Lipman et al. https://arxiv.org/pdf/2210.02747
         B) Linear with dynamics as data prediction VF = x1 - x0 --> x_next = xt +  dt * (x_hat - x0) see Tong et al. https://arxiv.org/pdf/2302.00482
        Both of which can add additional noise.
        """
        # x_next = xt + self.update_weight(t) * dt * (x_hat - xt)
        if x0 is None:
            if last_step:  #! this is what happens when T = 1 since xt's cancel
                return x_hat  # xt + timesteps*1/timesteps * (x_hat - xt)
            data_scale, noise_scale = self.reverse_schedule(batch, time, dt)
            x_next = data_scale * x_hat + noise_scale * xt
        else:
            data_scale, _ = self.reverse_schedule(batch, time, dt)
            x_next = xt + data_scale * (x_hat - x0)
        if self.noise_sigma > 0:
            x_next += self.prior(batch, x_hat.shape, x_hat.device) * self.noise_sigma  # torch.randn_like(x_hat)
        return x_next


class DiscreteDiffusionInterpolant(Interpolant):
    """
    Class for continuous interpolation.
    Note the udnerlying D3PM only works for discrete time.
    Can look into MultiFlow Precurssor that uses CTMC for continuous time discrete diffusion (https://arxiv.org/pdf/2205.14987, https://github.com/andrew-cr/tauLDR/blob/main/lib/models/models.py)
    Argmax Flow also operate over discrete time
    Continuous time can work for cintuous gaussian representations found in DiffSBDD but we are not doing this.

    Attributes:
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        prior_type: str = "uniform",
        diffusion_type: str = 'd3pm',
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = 'discrete',
        num_classes: int = 12,
        custom_prior: torch.Tensor = None,
        scheduler_type='cosine_adaptive',
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
        cut: bool = True,
    ):
        super(DiscreteDiffusionInterpolant, self).__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.custom_prior = custom_prior
        self.diffusion_type = diffusion_type
        self.init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip, cut)

    def get_Qt(self, terminal_distribution: torch.Tensor):
        #! If terminal distriubtion is [0 ... 0, 1] then we get masking state
        #! If terminal is [1/k ... 1/k] we get uniform
        # See Appendix A.2 D3PM https://arxiv.org/pdf/2107.03006
        QT = []
        for alpha_t in self.alphas:
            stay_prob = torch.eye(len(terminal_distribution)) * alpha_t
            diffuse_prob = (1.0 - alpha_t) * (
                torch.ones(1, len(terminal_distribution)) * (terminal_distribution.unsqueeze(0))
            )
            QT.append(stay_prob + diffuse_prob)
        return torch.stack(QT, dim=0)

    def d3pm_setup(self):
        self.discrete_time_only = True
        if self.prior_type == "uniform":
            prior_dist = torch.ones((self.num_classes)) * 1 / self.num_classes
        elif self.prior_type in ["absorb", "mask"]:
            prior_dist = torch.zeros((self.num_classes))
            prior_dist[-1] = 1.0
        elif self.prior_type in ["custom", "data"]:
            prior_dist = self.custom_prior
        assert torch.sum(prior_dist).item() - 1.0 < 1e-5
        Qt = self.get_Qt(prior_dist)
        Qt_prev = torch.eye(self.num_classes)
        Qt_bar = []
        for i in range(len(self.alphas)):
            Qtb = Qt_prev @ Qt[i]
            Qt_bar.append(Qtb)
            Qt_prev = Qtb
        Qt_bar = torch.stack(Qt_bar)
        Qt_bar_prev = Qt_bar[:-1]
        Qt_prev_pad = torch.eye(self.num_classes)
        Qt_bar_prev = torch.concat([Qt_prev_pad.unsqueeze(0), Qt_bar_prev], dim=0)

        return Qt, Qt_bar, Qt_bar_prev

    def init_schedulers(self, timesteps, scheduler_type, s, sqrt, nu, clip, cut):
        self.schedule_type = scheduler_type
        self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip, cut)
        alphas, betas = self.scheduler.get_alphas_and_betas()
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        alpha_bar = torch.exp(log_alpha_bar)
        # alpha_bar_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        if self.diffusion_type == "d3pm":  # MIDI and EQ
            Qt, Qt_bar, Qt_prev_bar = self.d3pm_setup()
            self.register_buffer('alphas', alphas[-self.timesteps :])
            self.register_buffer('alpha_bar', alpha_bar[-self.timesteps :])
            self.register_buffer('forward_data_schedule', Qt_bar[-self.timesteps :])
            self.register_buffer('forward_noise_schedule', Qt_bar[-self.timesteps :])
            self.register_buffer('Qt', Qt[-self.timesteps :])
            self.register_buffer('Qt_bar', Qt_bar[-self.timesteps :])
            self.register_buffer('Qt_prev_bar', Qt_prev_bar[-self.timesteps :])
        elif self.diffusion_type == "argmax":  # TargetDiff
            self.register_buffer('forward_data_schedule', log_alpha_bar)
            self.register_buffer('forward_noise_schedule', log_1_min_a(log_alpha_bar))

    def forward_schedule(self, batch, time):
        # t = 1 - t
        t_idx = self.timesteps - 1 - time
        return self.forward_data_schedule[t_idx][batch], self.forward_noise_schedule[t_idx][batch]

    def reverse_schedule(self, time):
        t_idx = self.timesteps - 1 - time
        return (
            self.reverse_data_schedule[t_idx].unsqueeze(1),
            self.reverse_noise_schedule[t_idx].unsqueeze(1),
            self.log_var[t_idx].unsqueeze(1),
        )

    def interpolate(self, batch, x1, time):
        """
        Interpolate using discrete interpolation method.
        """
        if len(x1.shape) == 1:
            x1_hot = F.one_hot(x1, self.num_classes)
        else:
            x1_hot = x1
        if self.diffusion_type == "d3pm":
            assert self.discrete_time_only
            ford = self.forward_schedule(batch, time)[0]
            probs = torch.einsum(
                "nj, nji -> ni", [x1_hot.float(), ford]
            )  #! Eqn 3 of D3PM https://arxiv.org/pdf/2107.03006
            assert torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
            xt = probs.multinomial(
                1,
            ).squeeze()
            return x1, xt, probs

        elif self.diffusion_type == "argmax":
            log_x0 = index_to_log_onehot(x1)
            data_scale, noise_scale = self.forward_schedule(time)
            log_probs = log_add_exp(log_x0 + data_scale, noise_scale - np.log(self.num_classes))
            xt = log_sample_categorical(log_probs)

        return x1, xt, None

    def interpolate_edges(self, batch, x1, x1_index, time):
        """
        Interpolate using discrete interpolation method.
        Similar to sample_edges_categorical https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/experiments/diffusion/categorical.py#L242
        """
        j, i = x1_index
        mask = j < i
        mask_i = i[mask]
        edge_attr_triu = x1[mask]
        _, edge_attr_t, upper_probs = self.interpolate(batch[mask_i], edge_attr_triu, time)
        edge_index_global_perturbed, edge_attr_global_perturbed = self.clean_edges(x1_index, edge_attr_t)
        return x1, edge_attr_global_perturbed, _

    def prior(self, batch, shape, device, one_hot=False):
        """
        Returns discrete index (num_samples,) or one hot if True (num_samples, num_classes)
        """
        num_samples = shape[0]
        if self.prior_type in ["absorb", "mask"]:
            x0 = torch.ones((num_samples,)).to(torch.int64) * (self.num_classes - 1)
        elif self.prior_type == "uniform":
            x0 = torch.randint(0, self.num_classes, (num_samples,)).to(torch.int64)
        elif self.prior_type in ["custom", "data"]:
            x0 = torch.multinomial(self.custom_prior, num_samples, replacement=True).to(torch.int64)
        else:
            raise ValueError("Only uniform and mask are supported")
        if one_hot:
            x0 = F.one_hot(x0, num_classes=self.num_classes)
        return x0.to(device)

    def prior_edges(self, batch, shape, index, device, one_hot=False, return_masks=False):
        """
        Returns discrete index (num_samples,) or one hot if True (num_samples, num_classes)
        similar to initialize_edge_attrs_reverse https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/experiments/diffusion/utils.py#L18
        """
        num_samples = shape[0]
        j, i = index
        mask = j < i
        mask_i = i[mask]
        num_upper_E = len(mask_i)
        num_samples = num_upper_E
        edge_attr_triu = self.prior(batch, (num_samples, self.num_classes), device)
        edge_index_global, edge_attr_global, mask, mask_i = self.clean_edges(index, edge_attr_triu, return_masks=True)
        if one_hot:
            edge_attr_global = F.one_hot(edge_attr_global, num_classes=self.num_classes).float()
        if return_masks:
            return edge_attr_global.to(device), edge_index_global.to(device), mask, mask_i
        else:
            return edge_attr_global.to(device), edge_index_global.to(device)

    def step(self, batch, xt, x_hat, time, x0=None, dt=None):
        """
        Perform a euler step in the discrete interpolant method.
        xt is the discrete variable at time t (or one hot)
        x_hat is the softmax of the logits
        """
        if self.solver_type == "sde" and self.diffusion_type == "d3pm":
            assert self.discrete_time_only
            # if len(x_hat.shape) <= 1:
            #     x_hat = F.one_hot(x_hat, num_classes=self.num_classes).float()
            if len(xt.shape) <= 1:
                xt = F.one_hot(xt, num_classes=self.num_classes).float()
            t_idx = self.timesteps - 1 - time[batch]

            # a = torch.einsum("nj, nji -> ni", [xt, self.Qt[t_idx].transpose(-2, -1)])
            # b = torch.einsum("nj, nji -> ni", [x_hat, self.Qt_prev_bar[t_idx]])
            # p0 = a * b
            # # (n, k)
            # p1 = torch.einsum("nj, nji -> ni", [x_hat, self.Qt_bar[t_idx]])
            # p1 = (p1 * xt).sum(-1, keepdims=True)
            # # (n, 1)

            # probs = p0 / p1
            # check = torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
            # assert check
            # ! The above is the exact same but uses one more einsum so probably slower
            A = torch.einsum("nj, nji -> ni", [xt, self.Qt[t_idx].permute(0, 2, 1)]).unsqueeze(1)  # [A, 1, 16]
            B = self.Qt_prev_bar[t_idx]  # [A, 16, 16]
            p0 = A * B
            p1 = torch.einsum("nij, nj -> ni", [self.Qt_bar[t_idx], xt]).unsqueeze(-1)
            probs = p0 / (p1.clamp(min=1e-5))
            unweighted_probs = (probs * x_hat.unsqueeze(-1)).sum(1)
            unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
            # (N, a_t-1)
            probs = unweighted_probs / (unweighted_probs.sum(-1, keepdims=True) + 1.0e-5)
            x_next = probs.multinomial(
                1,
            ).squeeze()
        else:
            raise ValueError("Only SDE Implemented for D3PM")

        return x_next

    def step_edges(
        self, batch, edge_index, edge_attr_t, edge_attr_hat, time, mask=None, mask_i=None, return_masks=False
    ):
        if mask is None or mask_i is None:
            j, i = edge_index
            mask = j < i
            mask_i = i[mask]
        edge_attr_t = edge_attr_t[mask]
        edge_attr_hat = edge_attr_hat[mask]
        edge_attr_next = self.step(batch[mask_i], edge_attr_t, edge_attr_hat, time)
        return self.clean_edges(edge_index, edge_attr_next, return_masks=return_masks)

    def clean_edges(self, edge_index, edge_attr_next, one_hot=False, return_masks=False):
        j, i = edge_index
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global = torch.stack([j, i], dim=0)
        edges_triu = F.one_hot(edge_attr_next, self.num_classes).float()
        edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            sort_by_row=False,
        )
        if not one_hot:
            edge_attr_global = edge_attr_global.argmax(1)
        if return_masks:
            return edge_index_global, edge_attr_global, mask, mask_i
        else:
            return edge_index_global, edge_attr_global

    def snr(self, time):
        t_idx = self.timesteps - 1 - time
        abar = self.alpha_bar[t_idx]
        return abar / (1 - abar)


class DiscreteFlowMatchingInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        # schedule_params: dict = {'type': 'linear', 'time': 'uniform', 'time_type': 'continuous'},
        prior_type: str = "uniform",
        update_weight_type: str = "constant",
        solver_type: str = "ode",
        timesteps: int = 500,
        min_t: float = 1e-2,
        time_type: str = 'continuous',
        num_classes: int = 10,
        custom_prior: torch.Tensor = None,
        scheduler_type='linear',
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
    ):
        super(DiscreteFlowMatchingInterpolant, self).__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.update_weight_type = update_weight_type
        self.min_t = min_t
        self.custom_prior = custom_prior
        self.init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip)

    def init_schedulers(self, timesteps, scheduler_type, s, sqrt, nu, clip):
        self.schedule_type = scheduler_type
        if scheduler_type == "linear":  #! vpe_linear is just linear with an update weight of recip_time_to_go
            self.discrete_time_only = False
            time = torch.linspace(self.min_t, 1, self.timesteps)
            self.register_buffer("time", time)
            self.register_buffer("forward_data_schedule", time)
            self.register_buffer("forward_noise_schedule", 1.0 - time)
        elif scheduler_type == "vpe":
            self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip)
            alphas, betas = self.scheduler.get_alphas_and_betas()
            # FlowMol defines alpha as 1 - cos ^2
            # self.forward_data_schedule = self.alphas
            # self.reverse_data_schedule = 1.0 - self.alphas
            # self.derivative_forward_data_schedule = self.alphas_prime
            # self.alpha_bar = self.alphas
            self.register_buffer('alphas', alphas)
            self.register_buffer('betas', betas)
            self.register_buffer('alpha_bar', alphas)
            self.register_buffer('forward_data_schedule', alphas)
            self.register_buffer('reverse_data_schedule', 1.0 - self.alphas)

    def snr_loss_weight(self, time):
        if self.time_type == "continuous":
            return torch.clamp(time / (1 - time), min=0.05, max=1.5)
        else:
            if self.schedule_type == "linear":
                t = time / self.timesteps
                return torch.clamp(t / (1 - t), min=0.05, max=1.5)
            else:
                return torch.clamp(self.snr(time), min=0.05, max=1.5)

    def update_weight(self, t):
        if self.update_weight_type == "constant":
            weight = torch.ones_like(t).to(t.device)
        elif self.update_weight_type == "recip_time_to_go":
            weight = torch.clamp(1 / (1 - t), max=self.timesteps)  # at T = 1 this makes data_scale = 1
        return weight

    def forward_schedule(self, batch, time):
        if self.time_type == "continuous":
            if self.schedule_type == "linear":
                return time[batch].unsqueeze(1), (1.0 - time)[batch].unsqueeze(1)
            else:
                raise NotImplementedError("Continuoys time is only implemented with linear schedule")
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

    def interpolate(self, batch, x1, time):
        """
        Interpolate using discrete interpolation method.
        """
        if self.prior_type in ["mask", "absorb", "uniform"]:
            x0 = self.prior(batch, x1.shape, self.num_classes, x1.device).unsqueeze(1)
            if self.time_type == "continuous":
                t = time
            else:
                t = time / self.timesteps
            t = t[batch]
            xt = x1.clone().unsqueeze(1)
            corrupt_mask = torch.rand((x1.shape[0], 1)).to(t.device) < (1 - t.unsqueeze(1))
            xt[corrupt_mask] = x0[corrupt_mask]
        else:
            raise ValueError("Only uniform and mask are supported")

        return x1, xt.squeeze(1), x0.squeeze(1)

    def prior(self, batch, shape, device, one_hot=False):
        """
        Returns discrete index (num_samples, 1) or one hot if True (num_samples, num_classes)
        """
        num_samples = shape[0]
        if self.prior_type in ["absorb", "mask"]:
            x0 = torch.ones((num_samples,)).to(torch.int64) * (self.num_classes - 1)
        elif self.prior_type == "uniform":
            x0 = torch.randint(0, self.num_classes, (num_samples,)).to(torch.int64)
        elif self.prior_type in ["custom", "data"]:
            x0 = torch.multinomial(self.custom_prior, num_samples, replacement=True).to(torch.int64)
        else:
            raise ValueError("Only uniform and mask/absorb are supported")
        if one_hot:
            x0 = F.one_hot(x0, num_classes=self.num_classes)
        return x0.to(device)

    def clean_edges(self, edge_index, edge_attr_next):
        assert False
        j, i = edge_index
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global = torch.stack([j, i], dim=0)
        edges_triu = F.one_hot(edge_attr_next, self.num_classes).float()
        edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            sort_by_row=False,
        )
        return edge_index_global, edge_attr_global, mask, mask_i

    def step(
        self,
        batch,
        xt,  #! if takes in one hot it will convert it
        x_hat,  #! assumes input is logits
        time,
        dt,
        x0=None,
        stochasticity=1,
        temp=0.1,
        use_purity=False,
        last_step=False,
    ):
        """
        Perform a euler step in the discrete interpolant method.
        """
        # TODO: Take a look at FlowMol since we can remove this last step stuff and clean_up the code can change it to if time == 1 then we do armax
        # TODO: take all arguments that are not x time and batch and set them up as class variables
        assert False
        if len(xt.shape) > 1:
            xt = xt.argmax(dim=-1)
        N = stochasticity
        S = self.num_classes
        MASK_TOKEN_INDEX = S - 1
        if self.time_type == "continuous":
            t = time
        else:
            t = time / self.timesteps

        t = t[batch].unsqueeze(1)
        if self.prior_type in ["uniform", "data", "custom"]:
            logits_1 = x_hat
            if last_step:
                x_next = torch.argmax(logits_1, dim=-1)
            else:
                pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)
                pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=xt.long().unsqueeze(-1))
                step_probs = dt * (pt_x1_probs * ((1 + N + N * (S - 1) * t) / (1 - t)) + N * pt_x1_eq_xt_prob)

                step_probs = self._regularize_step_probs(step_probs, xt)
                x_next = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(xt.shape)  # Same as categorical
        elif self.prior_type in ["mask", "absorb"]:
            #! Masking is initalized with one more column as the mask state
            logits_1 = x_hat.clone()
            device = logits_1.device
            if last_step:
                logits_1[:, MASK_TOKEN_INDEX] = -1e9
                x_next = torch.argmax(logits_1, dim=-1)
            else:
                if use_purity:
                    x_next = self.discrete_purity_step(dt, t, logits_1, xt, batch, noise=stochasticity, temp=temp)
                else:
                    mask_one_hot = torch.zeros((S,), device=device)
                    mask_one_hot[MASK_TOKEN_INDEX] = 1.0

                    logits_1[:, MASK_TOKEN_INDEX] = -1e9

                    pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)

                    xt_is_mask = (xt == MASK_TOKEN_INDEX).view(-1, 1).float()
                    step_probs = dt * pt_x1_probs * ((1 + N * t) / ((1 - t)))  #!UNMASK
                    step_probs += dt * (1 - xt_is_mask) * mask_one_hot.view(1, -1) * N  #!MASK UNMASKED STATES

                    step_probs = self._regularize_step_probs(step_probs, xt)

                    x_next = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(
                        xt.shape
                    )  # Same as categorical

        return x_next

    def _regularize_step_probs(self, step_probs, aatypes_t):
        #! TODO look into if Batch matters here but should not since everything is -1 so over atom classes
        num_res, S = step_probs.shape
        device = step_probs.device
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        # TODO replace with torch._scatter
        step_probs[torch.arange(num_res, device=device), aatypes_t.long().flatten()] = 0
        step_probs[torch.arange(num_res, device=device), aatypes_t.long().flatten()] = (
            1.0 - torch.sum(step_probs, dim=-1).flatten()
        )
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        return step_probs

    def discrete_purity_step(self, d_t, t, logits_1, aatypes_t, batch_ligand, noise=5, temp=0.1):
        pass
        # num_res, S = logits_1.shape

        # assert aatypes_t.shape == (num_res, 1)
        # assert S == self.num_classes
        # device = logits_1.device
        # MASK_TOKEN_INDEX = S-1

        # logits_1_wo_mask = logits_1[:, 0:-1] # (D, S-1)
        # pt_x1_probs = F.softmax(logits_1_wo_mask / temp, dim=-1) # (B, D, S-1)
        # # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
        # max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
        # # bias so that only currently masked positions get chosen to be unmasked
        # max_logprob = max_logprob - (aatypes_t != MASK_TOKEN_INDEX).float() * 1e9
        # sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)

        # unmask_probs = (d_t * ( (1 + noise * t) / (1-t)).to(device)).clamp(max=1) # scalar

        # number_to_unmask = torch.binomial(count=torch.count_nonzero(aatypes_t == MASK_TOKEN_INDEX, dim=-1).float(),prob=unmask_probs)
        # unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S-1), num_samples=1).view(num_res, 1)

        #! TODO figure out how to do this for no batch size
        # D_grid = torch.arange(num_res, device=device).view(1, -1) #.repeat(batch_size, 1)
        # mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        # inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1) #.repeat(1, num_res)
        # masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
        # mask2 = torch.zeros((num_res, 1), device=device)
        # mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((num_res, 1), device=device))
        # unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        # mask2 = mask2 * (1 - unmask_zero_row)
        # aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2

        # # re-mask
        # u = torch.rand(batch_size, num_res, device=device) #! Need to have the ligand index
        # re_mask_mask = (u < d_t * noise).float()
        # aatypes_t = aatypes_t * (1 - re_mask_mask) + MASK_TOKEN_INDEX * re_mask_mask

        # return aatypes_t


def test_continuous_diffusion(ligand_pos, batch_ligand):
    print("ContinuousDiffusionInterpolant")
    pos_interpolant = ContinuousDiffusionInterpolant()
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='continuous diffusion interpolation', total=len(time_seq)):
        t_idx = pos_interpolant.sample_time(4, method='stab_mode')
        data_scale, noise_scale = pos_interpolant.forward_schedule(t_idx, batch_ligand)
        x1, xt, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t_idx=t_idx, com_free=True)

    xt = pos_interpolant.prior(x1.shape, batch_ligand, com_free=True, device=x1.device)
    for i in tqdm(time_seq, desc='continuous diffusion step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t_idx=t_idx, com_free=True)
        x_hat = xt01
        x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t_idx)
        xt = x_tp1


def test_continuous_flowmatching(ligand_pos, batch_ligand):
    print("ContinuousFlowMatchingInterpolant")
    pos_interpolant = ContinuousFlowMatchingInterpolant(update_weight_type='recip_time_to_go')
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='continuous direct time interpolation', total=len(time_seq)):
        t = pos_interpolant.sample_time(4, method='uniform', min_t=pos_interpolant.min_t)
        data_scale, noise_scale = pos_interpolant.forward_schedule(t=t, batch=batch_ligand)
        x1, xt, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t=t, com_free=True)
    pos_interpolant = ContinuousFlowMatchingInterpolant(
        update_weight_type='recip_time_to_go',
        time_type='discrete',
    )
    for i in tqdm(time_seq, desc='continuous time_idx interpolation', total=len(time_seq)):
        t_idx = pos_interpolant.sample_time(4, method='uniform')
        data_scale, noise_scale = pos_interpolant.forward_schedule(t_idx=t_idx, batch=batch_ligand)
        x1, xt, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t_idx=t_idx, com_free=True)

    xt = pos_interpolant.prior(x1.shape, batch_ligand, com_free=True, device=x1.device)
    dt = 1 / 500
    check = [torch.sum((ligand_pos - xt) ** 2)]
    for i in tqdm(time_seq, desc='continuous time_index flow step static dt', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t_idx=t_idx, com_free=True)
        x_hat = x1 + 0.0001 * pos_interpolant.prior(x1.shape, batch_ligand, True, x1.device)
        x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t_idx=t_idx, dt=dt)
        xt = x_tp1
        check.append(torch.sum((ligand_pos - xt) ** 2))

    pos_interpolant = ContinuousFlowMatchingInterpolant(update_weight_type='recip_time_to_go')
    xt = pos_interpolant.prior(x1.shape, batch_ligand, com_free=True, device=x1.device)
    dt = 1 / 500
    check = [torch.sum((ligand_pos - xt) ** 2)]
    time_seq = torch.linspace(1e-2, 1, 500)  # min_t used in multi flow
    for i in tqdm(time_seq, desc='continuous time flow step static dt', total=len(time_seq)):
        t = torch.full(size=(4,), fill_value=i, device='cpu')
        x1, xt01, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t=t, com_free=True)
        x_hat = x1 + 0.0001 * pos_interpolant.prior(x1.shape, batch_ligand, True, x1.device)
        x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t=t, dt=dt)
        xt = x_tp1
        check.append(torch.sum((ligand_pos - xt) ** 2))

    #! linspace is inclusing just raw discrete time*timeesteps is not this is a discrepancy that is not noticable in diffusion when everything is indexed.
    # ! We are going to follow MultiFlow for this so we add a clamp to the forward schedule for the t = 1 pass
    xt = pos_interpolant.prior(x1.shape, batch_ligand, com_free=True, device=x1.device)
    check = [torch.sum((ligand_pos - xt) ** 2)]
    time_seq = torch.linspace(1e-2, 1, 500)  # min_t used in multi flow
    for i in tqdm(range(1, len(time_seq) + 1), desc='continuous time flow step dynamic dt', total=len(time_seq)):
        t = torch.full(size=(4,), fill_value=time_seq[i - 1], device='cpu')
        if i < len(time_seq):
            t_next = torch.full(size=(4,), fill_value=time_seq[i], device='cpu')

        x1, xt01, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t=t, com_free=True)
        x_hat = x1 + 0.0001 * pos_interpolant.prior(x1.shape, batch_ligand, True, x1.device)
        if i == len(time_seq):
            x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t=t, last_step=True)
        else:
            x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t=t, t_next=t_next)
        xt = x_tp1
        check.append(torch.sum((ligand_pos - xt) ** 2))
    x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t=t, t_next=t_next)


def test_discrete_diffusion(h, batch):
    print("DiscreteDiffusionInterpolant-Uniform: ONLY WORKS with DISCRETE TIME")
    num_classes = 13

    interpolant = DiscreteDiffusionInterpolant(num_classes=num_classes)
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete diffusion interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch)
        x1, xt, probs = interpolant.interpolate(h, batch, t_idx=t_idx)
    assert all(torch.argmax(probs, 1) == x1)

    xt = interpolant.prior(h.shape, batch, device=h.device)
    for i in tqdm(time_seq, desc='discrete diffusion step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, probs = interpolant.interpolate(h, batch, t_idx=t_idx)
        x_hat = xt01
        x_tp1 = interpolant.step(xt, x_hat, batch, t_idx)
        xt = x_tp1

    print("DiscreteDiffusionInterpolant-Absorb")
    num_classes = 14

    interpolant = DiscreteDiffusionInterpolant(num_classes=num_classes, prior_type="absorb")
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete diffusion absorb interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch)
        x1, xt, probs = interpolant.interpolate(h, batch, t_idx=t_idx)
    probs[:, -1] = 0
    assert all(torch.argmax(probs, 1) == x1)

    xt = interpolant.prior(h.shape, batch, device=h.device)
    for i in tqdm(time_seq, desc='discrete diffusion absorb step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, probs = interpolant.interpolate(h, batch, t_idx=t_idx)
        x_hat = xt01
        x_tp1 = interpolant.step(xt, x_hat, batch, t_idx)
        xt = x_tp1


def test_discrete_flowmatching(h, batch):
    print("DiscreteFlowMatchingInterpolant-Uniform")
    num_classes = 13

    interpolant = DiscreteFlowMatchingInterpolant(
        num_classes=num_classes,
        prior_type="uniform",
        # schedule_params={'type': 'linear', 'time': 'uniform', 'time_type': 'discrete'},
        time_type='discrete',
    )
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete flowmatching interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch)
        x1, xt, x0 = interpolant.interpolate(h, batch, t_idx=t_idx)

    xt = interpolant.prior(h.shape, batch, device=h.device)
    for i in tqdm(time_seq, desc='discrete diffusion step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, x0 = interpolant.interpolate(h, batch, t_idx=t_idx)
        x_hat = F.one_hot(xt01, num_classes)
        x_tp1 = interpolant.step(xt, x_hat, batch, t_idx=t_idx, dt=1 / 500)
        xt = x_tp1

    interpolant = DiscreteFlowMatchingInterpolant(
        num_classes=num_classes,
        prior_type="uniform",
        # schedule_params={'type': 'linear', 'time': 'uniform', 'time_type': 'continuous'},
        time_type='continuous',
    )
    xt = interpolant.prior(h.shape, batch, device=h.device)
    dt = 1 / 500
    time_seq = torch.linspace(1e-2, 1, 500)  # min_t used in multi flow
    for i in tqdm(time_seq, desc='continuous time discrete flow step static dt', total=len(time_seq)):
        t = torch.full(size=(4,), fill_value=i, device='cpu')
        x1, xt01, x0 = interpolant.interpolate(h, batch, t=t)
        x_hat = F.one_hot(x1, num_classes)
        if i == 1.0:
            x_tp1 = interpolant.step(xt, x_hat, batch, t=t, dt=dt, last_step=True)
        else:
            x_tp1 = interpolant.step(xt, x_hat, batch, t=t, dt=dt)
        xt = x_tp1

    xt = interpolant.prior(h.shape, batch, device=h.device)
    time_seq = torch.linspace(1e-2, 1, 500)  # min_t used in multi flow
    for i in tqdm(
        range(1, len(time_seq) + 1), desc='continuous time discrete flow step dynamic dt', total=len(time_seq)
    ):
        t = torch.full(size=(4,), fill_value=time_seq[i - 1], device='cpu')
        if i < len(time_seq):
            t_next = torch.full(size=(4,), fill_value=time_seq[i], device='cpu')
        x1, xt01, x0 = interpolant.interpolate(h, batch, t=t)
        x_hat = F.one_hot(x1, num_classes)
        if i == len(time_seq):
            x_tp1 = interpolant.step(xt, x_hat, batch, t=t, dt=1 / 500, last_step=True)
        else:
            x_tp1 = interpolant.step(xt, x_hat, batch, t=t, t_next=t_next)
        xt = x_tp1
    assert all(x1 == xt)

    print("DiscreteDiffusionInterpolant-Absorb")
    num_classes = 14

    interpolant = DiscreteFlowMatchingInterpolant(
        num_classes=num_classes,
        prior_type="absorb",
        # schedule_params={'type': 'linear', 'time': 'uniform', 'time_type': 'discrete'},
        time_type='discrete',
    )
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete diffusion absorb interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch)
        x1, xt, x0 = interpolant.interpolate(h, batch, t_idx=t_idx)

    xt = interpolant.prior(h.shape, batch, device=h.device)
    for i in tqdm(time_seq, desc='discrete diffusion absorb step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, x0 = interpolant.interpolate(h, batch, t_idx=t_idx)
        x_hat = F.one_hot(xt01, num_classes)
        x_tp1 = interpolant.step(xt, x_hat, batch, t_idx=t_idx, dt=1 / 500)
        xt = x_tp1

    interpolant = DiscreteFlowMatchingInterpolant(
        num_classes=num_classes,
        prior_type="absorb",
        # schedule_params={'type': 'linear', 'time': 'uniform', 'time_type': 'continuous'},
        time_type='continuous',
    )

    xt = interpolant.prior(h.shape, batch, device=h.device)
    dt = 1 / 500
    time_seq = torch.linspace(1e-2, 1, 500)  # min_t used in multi flow
    for i in tqdm(time_seq, desc='continuous time discrete flow  absorb step static dt', total=len(time_seq)):
        t = torch.full(size=(4,), fill_value=i, device='cpu')
        x1, xt01, x0 = interpolant.interpolate(h, batch, t=t)
        x_hat = F.one_hot(x1, num_classes)
        if i == 1.0:
            x_tp1 = interpolant.step(xt, x_hat, batch, t=t, dt=dt, last_step=True)
        else:
            x_tp1 = interpolant.step(xt, x_hat, batch, t=t, dt=dt)
        xt = x_tp1

    xt = interpolant.prior(h.shape, batch, device=h.device)
    time_seq = torch.linspace(1e-2, 1, 500)  # min_t used in multi flow
    for i in tqdm(
        range(1, len(time_seq) + 1), desc='continuous time discrete flow step dynamic dt', total=len(time_seq)
    ):
        t = torch.full(size=(4,), fill_value=time_seq[i - 1], device='cpu')
        if i < len(time_seq):
            t_next = torch.full(size=(4,), fill_value=time_seq[i], device='cpu')
        x1, xt01, x0 = interpolant.interpolate(h, batch, t=t)
        x_hat = F.one_hot(x1, num_classes)
        if i == len(time_seq):
            x_tp1 = interpolant.step(xt, x_hat, batch, t=t, dt=1 / 500, last_step=True)
        else:
            x_tp1 = interpolant.step(xt, x_hat, batch, t=t, t_next=t_next)
        xt = x_tp1
    assert all(x1 == xt)


def test_discrete_edge_diffusion(E, E_idx, batch):
    print("DiscreteDiffusionInterpolant-Uniform: ONLY WORKS with DISCRETE TIME")
    num_classes = 5

    interpolant = DiscreteDiffusionInterpolant(num_classes=num_classes)
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete diffusion interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch)
        x1, xt, probs = interpolant.interpolate_edges(E, E_idx, batch, t_idx=t_idx)

    xt, xt_index = interpolant.prior_edges(E.shape, E_idx, batch, device=E.device)
    # MASK = mask.clone()
    for i in tqdm(time_seq, desc='discrete diffusion step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, probs = interpolant.interpolate_edges(E, E_idx, batch, t_idx=t_idx)
        x_hat = xt01
        xt_index, x_tp1 = interpolant.step_edges(
            xt_index, xt, x_hat, batch, t_idx
        )  #! The mask's do not change its alwasy fully connected here
        xt = x_tp1

    num_classes = 6

    interpolant = DiscreteDiffusionInterpolant(num_classes=num_classes, prior_type="absorb")
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete diffusion interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch)
        x1, xt, probs = interpolant.interpolate_edges(E, E_idx, batch, t_idx=t_idx)

    xt, xt_index = interpolant.prior_edges(E.shape, E_idx, batch, device=E.device)
    for i in tqdm(time_seq, desc='discrete diffusion step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, probs = interpolant.interpolate_edges(E, E_idx, batch, t_idx=t_idx)
        x_hat = xt01
        xt_index, x_tp1 = interpolant.step_edges(xt_index, xt, x_hat, batch, t_idx)
        xt = x_tp1


if __name__ == "__main__":
    from tqdm import tqdm

    print("TODO fix tests after refactor of the time element and batch first arg calling")
    # assert False
    # print("TEST")
    # ligand_pos = torch.rand((75, 3))
    # batch_ligand = torch.Tensor(
    #     [
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         2,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #         3,
    #     ]
    # ).to(torch.int64)
    # ligand_feats = torch.Tensor(
    #     [
    #         2,
    #         4,
    #         2,
    #         4,
    #         2,
    #         4,
    #         4,
    #         3,
    #         2,
    #         2,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         5,
    #         1,
    #         3,
    #         1,
    #         1,
    #         1,
    #         2,
    #         4,
    #         2,
    #         4,
    #         2,
    #         4,
    #         4,
    #         3,
    #         2,
    #         2,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         5,
    #         1,
    #         3,
    #         1,
    #         1,
    #         1,
    #         2,
    #         2,
    #         2,
    #         2,
    #         12,
    #         2,
    #         5,
    #         2,
    #         3,
    #         5,
    #         1,
    #         5,
    #         2,
    #         4,
    #         2,
    #         4,
    #         2,
    #         4,
    #         4,
    #         3,
    #         2,
    #         2,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         5,
    #         1,
    #         3,
    #         1,
    #         1,
    #         1,
    #     ]
    # ).to(torch.int64)

    # num_classes = 13
    # # Initialize the adjacency matrix with zeros
    # adj_matrix = torch.zeros((75, 75, 5), dtype=torch.int64)
    # no_bond = torch.zeros(5)
    # no_bond[0] = 1
    # # Using broadcasting to create the adjacency matrix
    # adj_matrix[batch_ligand.unsqueeze(1) == batch_ligand] = 1
    # for idx, i in enumerate(batch_ligand):
    #     for jdx, j in enumerate(batch_ligand):
    #         if idx == jdx:
    #             adj_matrix[idx][jdx] = no_bond
    #         elif i == j:
    #             adj_matrix[idx][jdx] = torch.nn.functional.one_hot(torch.randint(0, 5, (1,)), 5).squeeze(0)
    # # print(adj_matrix)

    # # atom_embedder = nn.Linear(num_classes, 64)
    # X = ligand_pos
    # H = F.one_hot(ligand_feats, num_classes).float()  # atom_embedder(F.one_hot(ligand_feats, num_classes).float())
    # A = adj_matrix
    # mask = batch_ligand.unsqueeze(1) == batch_ligand.unsqueeze(0)  # Shape: (75, 75)
    # E_idx = mask.nonzero(as_tuple=False).t()
    # self_loops = E_idx[0] != E_idx[1]
    # E_idx = E_idx[:, self_loops]

    # source, target = E_idx
    # E = A[source, target]  # E x 5
    # E = E.argmax(1)  # E
    # E_idx, E = sort_edge_index(
    #     edge_index=E_idx,
    #     edge_attr=E,
    #     sort_by_row=False,
    # )
    # # edge_embedder = nn.Linear(5, 32)
    # # E = edge_embedder(E.float())
    # test_discrete_edge_diffusion(E, E_idx, batch_ligand)
    # test_continuous_diffusion(ligand_pos, batch_ligand)
    # test_continuous_flowmatching(ligand_pos, batch_ligand)
    # test_discrete_diffusion(ligand_feats, batch_ligand)
    # test_discrete_flowmatching(ligand_feats, batch_ligand)
    # print("SUCCESS")
