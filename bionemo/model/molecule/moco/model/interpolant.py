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
from torch_scatter import scatter_mean


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


def cosine_beta_schedule(params, num_diffusion_timesteps, s=0.008, nu=1.0, sqrt=False, return_alpha=False):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = num_diffusion_timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos((((x / steps) ** nu) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.clip(alphas, a_min=0.001, a_max=1.0)  #! should this be 0.999 like EQ
    if sqrt:
        alphas = np.sqrt(alphas)
    if return_alpha:
        return alphas, 1 - alphas
    return 1 - alphas


def cosine_beta_schedule_fm(params, num_diffusion_timesteps, s=0.008, nu=1.0):
    """
    cosine schedule
    as proposed in FlowMol
    """
    steps = num_diffusion_timesteps + 1
    x = np.linspace(0, steps, steps)
    t = x / steps
    alphas = 1 - np.cos((t**nu + s) / (1 + s) * np.pi * 0.5) ** 2
    t = torch.clamp_(t, min=1e-9)
    alpha_prime = np.sin(np.pi * (t + s) ** nu / (1 + s)) * (np.pi / 2) * (nu * (t + s) ** (nu - 1)) / (1 + s)
    return alphas, alpha_prime


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule_eq(params, num_diffusion_timesteps, s=0.008, nu=1.0):
    steps = num_diffusion_timesteps + 2
    x = torch.linspace(0, num_diffusion_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    ### new included
    alphas_cumprod = torch.from_numpy(clip_noise_schedule(alphas_cumprod, clip_value=0.05))
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = alphas.clip(min=0.001)
    betas = 1 - alphas
    betas = betas[1:]  #! Cut the extra piece that EQ skipped so we can iterate [0, 499] instead of [1, 500]
    betas = torch.clip(betas, 0.0, 0.999).float()
    return 1 - betas, betas


class Interpolant:
    def __init__(
        self,
        method_type: str,
        schedule_params: dict,
        prior_type: str,
        update_weight_type: str = "linear",
        solver_type: str = "ode",
        timesteps: int = 500,
    ):
        self.method_type = method_type
        self.schedule_params = schedule_params
        self.prior_type = prior_type
        self.update_weight_type = update_weight_type
        self.timesteps = timesteps
        self.solver_type = solver_type

    def sample_time_idx(self, num_samples, device, method, mean=0, scale=0.81):
        if method == 'symmetric':
            time_step = torch.randint(0, self.num_timesteps, size=(num_samples // 2 + 1,))
            time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_samples]

        elif method == 'uniform':
            time_step = torch.randint(0, self.num_timesteps, size=(num_samples,))
        elif method == "stab_mode":  #! converts uniform to Stability AI mode distribution

            def fmode(u: torch.Tensor, s: float) -> torch.Tensor:
                return 1 - u - s * (torch.cos((torch.pi / 2) * u) ** 2 - 1 + u)

            fmode(torch.rand(num_samples), scale)
        elif (
            method == 'logit_normal'
        ):  # see Figure 11 https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf
            torch.sigmoid(torch.normal(mean=mean, std=scale, size=(num_samples,)))
        else:
            raise ValueError
        return time_step.to(device)

    def update_weight(self, t):
        if self.update_weight_type == "constant":
            weight = 1
        elif self.update_weight_type == "recip_time_to_go":
            weight = 1 / (1 - t)
        return weight

    def snr(self, t_idx):
        abar = self.alpha_bar[t_idx]
        return abar / (1 - abar)

    def snr_loss_weight(self, t_idx):
        # return min(0.05, max(1.5, self.snr(t_idx)))
        return torch.clamp(self.snr(t_idx), min=0.05, max=1.5)


class ContinuousInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        method_type (str): diffusion or flow_matching
        schedule_params (dict): Type of interpolant schedule.
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        method_type: str,
        schedule_params: dict,
        prior_type: str,
        update_weight_type: str = "linear",
        solver_type: str = "ode",
        timesteps: int = 500,
    ):
        super(ContinuousInterpolant, self).__init__(
            method_type, schedule_params, prior_type, update_weight_type, solver_type, timesteps
        )
        self.init_schedulers(method_type, schedule_params, timesteps)

    def init_schedulers(self, method_type, schedule_params, timesteps):
        if method_type == "diffusion":
            self.alphas, self.betas = cosine_beta_schedule(schedule_params, timesteps, return_alphas=True)
            log_alpha = torch.log(self.alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            # self.log_alpha_bar = log_alpha_bar
            self.alpha_bar = alphas_cumprod = torch.exp(log_alpha_bar)
            self.alpha_bar_prev = alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            # sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            # sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).clamp(min=1e-4)
            self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            self.posterior_mean_c0_coef = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            self.posterior_mean_ct_coef = (1.0 - alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - alphas_cumprod)
            # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
            self.posterior_logvar = torch.log(
                torch.nn.functional.pad(self.posterior_variance[:-1], (1, 0), value=self.posterior_variance[0])
            )

            self.forward_data_schedule = torch.sqrt(self.alpha_bar)
            self.forward_noise_schedule = torch.sqrt(1 - self.alpha_bar)
            self.reverse_data_schedule = self.posterior_mean_c0_coef
            self.reverse_noise_schedule = self.posterior_mean_ct_coef
            self.log_var = self.posterior_logvar
        elif method_type == "flow_matching":
            if (
                schedule_params['type'] == "linear"
            ):  #! vpe_linear is just linear with an update weight of recip_time_to_go
                time = torch.linspace(0, 1, steps=timesteps)
                self.forward_data_schedule = time
                self.forward_noise_schedule = 1.0 - time
            elif schedule_params['type'] == "vpe":
                self.alphas, self.alphas_prime = cosine_beta_schedule_fm(
                    schedule_params, timesteps
                )  # FlowMol defines alpha as 1 - cos ^2
                self.forward_data_schedule = self.alphas
                self.reverse_data_schedule = 1.0 - self.alphas
                self.derivative_forward_data_schedule = self.alphas_prime
                self.alpha_bar = self.alphas  # For SNR

    def forward_schedule(self, t_idx):
        if self.method_type == 'diffusion':
            # t = 1 - t
            t_idx = self.timesteps - 1 - t_idx
        return self.forward_data_schedule[t_idx], self.forward_noise_schedule[t_idx]

    def reverse_schedule(self, t_idx, t=None, t_next=None, dt=None):
        if self.method_type == 'diffusion':
            t_idx = self.timesteps - 1 - t_idx
            return self.reverse_data_schedule[t_idx], self.reverse_noise_schedule[t_idx], self.log_var[t_idx]

        elif self.method_type == 'flow_matching':
            if dt is None:
                dt = t_next - t
            if self.schedule_params['type'] == "linear":
                data_scale = self.update_weight(t) * dt
            elif self.schedule_params['type'] == "vpe":  # FlowMol
                data_scale = (
                    self.derivative_forward_data_schedule[t_idx] * dt / (1 - self.forward_data_schedule[t_idx])
                )  # alpha_prime[t]*dt/(1 - alpha[t]) #! EquiFm uses (1-a)^2 could be due to the definition of the scheduler FloMol uses cosine wheres EquiFm uses exp(- 0.5 * integral of betas(s)) where beta is some noise scheduler funtion

            return data_scale, 1 - data_scale, None

    def interpolate(self, x1, t_idx, batch, com_free=True):
        """
        Interpolate using continuous flow matching method.
        """
        x0 = self.prior(x1.shape, batch, com_free).to(x1.device)
        data_scale, noise_scale = self.forward_schedule(t_idx)
        return x1, data_scale * x1 + noise_scale * x0, x0

    def prior(self, shape, batch, com_free, device):
        if self.prior_type == "gaussian" or self.prior_type == "normal":
            x0 = torch.randn(shape)
            if com_free:
                x0 = x0 - scatter_mean(x0, batch, dim=0)[batch]
        else:
            raise ValueError("Only Gaussian is supported")
        return x0.to(device)

    def step(self, xt, x_hat, batch, t_idx=None, t=None, dt=None, t_next=None, mask=None):
        """
        Perform a euler step in the continuous flow matching method.
        """
        if self.method_type == "diffusion":
            if self.solver_type == "sde":
                data_scale, noise_scale, log_var = self.reverse_schedule(t_idx, t, t_next, dt)
                # data_scale = extract(self.posterior_mean_c0_coef, t, batch)
                # noise_scale = extract(self.posterior_mean_ct_coef, t, batch)
                # pos_log_variance = extract(self.posterior_logvar, t, batch)
                mean = data_scale * x_hat + noise_scale * xt
                # no noise when diffusion t == 0 so flow matching t == 1
                nonzero_mask = (t == 1).float()[batch].unsqueeze(-1)
                # ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                x_next = mean + nonzero_mask * (0.5 * log_var).exp() * self.prior(xt.shape, batch, com_free=True)
                # TODO: can add guidance module here
            else:
                raise ValueError("Only SDE Implemented")
        elif self.method_type == "flow_matching":
            # x_next = xt + self.update_weight(t) * dt * (x_hat - xt)
            data_scale, noise_scale, _ = self.reverse_schedule(t_idx, t, t_next, dt)
            x_next = data_scale * x_hat + noise_scale * xt

        return x_next


class DiscreteInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        method_type (str): diffusion or flow_matching
        schedule_params (dict): Type of interpolant schedule.
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        method_type: str,
        schedule_params: dict,
        prior_type: str,
        update_weight_type: str = "linear",
        solver_type: str = "ode",
        timesteps: int = 500,
    ):
        super(DiscreteInterpolant, self).__init__(
            method_type, schedule_params, prior_type, update_weight_type, solver_type, timesteps
        )
        self.init_schedulers(method_type, schedule_params, timesteps)

    def init_schedulers(self, method_type, schedule_params, timesteps):
        if method_type == "diffusion":
            pass
        elif method_type == "flow_matching":
            if (
                schedule_params['type'] == "linear"
            ):  #! vpe_linear is just linear with an update weight of recip_time_to_go
                time = torch.linspace(0, 1, steps=timesteps)
                self.forward_data_schedule = time
                self.forward_noise_schedule = 1.0 - time
            elif schedule_params['type'] == "vpe":
                self.alphas, self.alphas_prime = cosine_beta_schedule_fm(
                    schedule_params, timesteps
                )  # FlowMol defines alpha as 1 - cos ^2
                self.forward_data_schedule = self.alphas
                self.reverse_data_schedule = 1.0 - self.alphas
                self.derivative_forward_data_schedule = self.alphas_prime
                self.alpha_bar = self.alphas

    def forward_schedule(self, t_idx):
        if self.method_type == 'diffusion':
            # t = 1 - t
            t_idx = self.timesteps - 1 - t_idx
        return self.forward_data_schedule[t_idx], self.forward_noise_schedule[t_idx]

    def reverse_schedule(self, t_idx, t=None, t_next=None, dt=None):
        if self.method_type == 'diffusion':
            t_idx = self.timesteps - 1 - t_idx
            return self.reverse_data_schedule[t_idx], self.reverse_noise_schedule[t_idx], self.log_var[t_idx]

        elif self.method_type == 'flow_matching':
            if dt is None:
                dt = t_next - t
            if self.schedule_params['type'] == "linear":
                data_scale = self.update_weight(t) * dt
            elif self.schedule_params['type'] == "vpe":  # FlowMol
                data_scale = (
                    self.derivative_forward_data_schedule[t_idx] * dt / (1 - self.forward_data_schedule[t_idx])
                )  # alpha_prime[t]*dt/(1 - alpha[t]) #! EquiFm uses (1-a)^2 could be due to the definition of the scheduler FloMol uses cosine wheres EquiFm uses exp(- 0.5 * integral of betas(s)) where beta is some noise scheduler funtion

            return data_scale, 1 - data_scale, None

    def interpolate(self, x1, t, num_classes):
        """
        Interpolate using discrete interpolation method.
        """
        if self.method_type == "diffusion":
            pass
        elif self.method_type == "flow_matching":
            if self.prior_type == "mask" or self.prior_type == "uniform":
                x0 = self.prior((x1.shape[0], num_classes), x1.device)
                t = pad_t_like_x(t, x0)
                xt = x1.clone()
                corrupt_mask = torch.rand((x1.shape[0], 1)).to(t.device) < (1 - t)  # [:, None])
                xt[corrupt_mask] = x0[corrupt_mask]
            else:
                raise ValueError("Only uniform and mask are supported")

        return x1, xt, x0

    def prior(self, shape, device, one_hot=False):
        """
        Returns discrete index (num_samples, 1) or one hot if True (num_samples, num_classes)
        """
        num_samples, num_classes = shape
        if self.prior_type == "mask":
            x0 = torch.ones((num_samples, 1)) * (num_classes - 1)
        elif self.prior_type == "uniform":
            x0 = torch.randint(0, num_classes, (num_samples, 1)).to(torch.int64)
        else:
            raise ValueError("Only uniform and mask are supported")
        if one_hot:
            x0 = F.one_hot(x0, num_classes=num_classes)
        return x0.to(device)

    def step(
        self,
        xt,
        x_hat,
        batch,
        t_idx=None,
        t=None,
        dt=None,
        t_next=None,
        mask=None,
        stochasticity=1,
        temp=0.1,
        use_purity=False,
        last_step=False,
    ):
        """
        Perform a euler step in the discrete interpolant method.
        """
        if self.method_type == "diffusion":
            if self.solver_type == "sde":
                x_next = None
            else:
                raise ValueError("Only SDE Implemented")
        elif self.method_type == "flow_matching":
            N = stochasticity
            S = x_hat.shape[-1]  # self.num_classes
            if self.prior_type == "uniform":
                logits_1 = x_hat
                pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)
                pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=xt.long().unsqueeze(-1))
                if last_step:
                    N = 0  #! no noise at final timestep
                step_probs = dt * (pt_x1_probs * ((1 + N + N * (S - 1) * t) / (1 - t)) + N * pt_x1_eq_xt_prob)

                step_probs = self._regularize_step_probs(step_probs, xt)
                x_next = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(xt.shape)  # Same as categorical
            elif self.prior_type == "mask":
                #! Masking is initalized with one more column as the mask state
                MASK_TOKEN_INDEX = S - 1
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

                        pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, D, S)

                        xt_is_mask = (xt == MASK_TOKEN_INDEX).view(-1, 1).float()
                        step_probs = dt * pt_x1_probs * ((1 + N * t) / ((1 - t)))  # (B, D, S) #!UNMASK
                        step_probs += dt * (1 - xt_is_mask) * mask_one_hot.view(1, -1) * N  #!MASK UNMASKED STATES

                        step_probs = self._regularize_step_probs(step_probs, xt)
                        x_next = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(
                            xt.shape
                        )  # Same as categorical

        return x_next

    def _regularize_step_probs(self, step_probs, aatypes_t):
        # batch_size, num_res, S = step_probs.shape
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
        # import ipdb; ipdb.set_trace()
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

        # import ipdb; ipdb.set_trace() #! TODO figure out how to do this for no batch size
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
