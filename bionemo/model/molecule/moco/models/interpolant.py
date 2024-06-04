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

from bionemo.model.molecule.moco.models.interpolant_utils import (
    float_time_to_index,
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_sample_categorical,
)
from bionemo.model.molecule.moco.models.scheduler import (
    cosine_beta_schedule,
    cosine_beta_schedule_eq,
    cosine_beta_schedule_fm,
)


class Interpolant:
    def __init__(
        self,
        schedule_params: dict,
        prior_type: str,
        solver_type: str = "sde",
        timesteps: int = 500,
    ):
        self.schedule_params = schedule_params
        self.prior_type = prior_type
        self.timesteps = timesteps
        self.solver_type = solver_type

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
        else:
            raise ValueError
        return time_step.to(device)

    def sample_time(self, num_samples, method, device='cpu', mean=0, scale=0.81, min_t=0):
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
        else:
            raise ValueError
        if min_t > 0:
            time_step = time_step * (1 - 2 * min_t) + min_t
        return time_step.to(device)

    def snr(self, t_idx):
        abar = self.alpha_bar[t_idx]
        return abar / (1 - abar)

    def snr_loss_weight(self, t_idx):
        # return min(0.05, max(1.5, self.snr(t_idx)))
        return torch.clamp(self.snr(t_idx), min=0.05, max=1.5)


class ContinuousDiffusionInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        schedule_params (dict): Type of interpolant schedule.
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        schedule_params: dict = None,
        prior_type: str = 'gaussian',
        solver_type: str = "sde",
        timesteps: int = 500,
    ):
        super(ContinuousDiffusionInterpolant, self).__init__(schedule_params, prior_type, solver_type, timesteps)
        self.init_schedulers(schedule_params, timesteps)

    def init_schedulers(self, schedule_params, timesteps):
        self.alphas, self.betas = cosine_beta_schedule_eq(
            schedule_params, timesteps
        )  # cosine_beta_schedule(schedule_params, timesteps, return_alpha=True)
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

    def forward_schedule(self, t_idx, batch):
        # t = 1 - t
        t_idx = self.timesteps - 1 - t_idx
        return (
            self.forward_data_schedule[t_idx].unsqueeze(1)[batch],
            self.forward_noise_schedule[t_idx].unsqueeze(1)[batch],
        )

    def reverse_schedule(self, t_idx, batch):
        t_idx = self.timesteps - 1 - t_idx
        return (
            self.reverse_data_schedule[t_idx].unsqueeze(1)[batch],
            self.reverse_noise_schedule[t_idx].unsqueeze(1)[batch],
            self.log_var[t_idx].unsqueeze(1)[batch],
        )

    def interpolate(self, x1, batch, t_idx, com_free=True):
        """
        Interpolate using continuous flow matching method.
        """
        x0 = self.prior(x1.shape, batch, com_free, x1.device)
        data_scale, noise_scale = self.forward_schedule(t_idx, batch)
        return x1, data_scale * x1 + noise_scale * x0, x0

    def prior(self, shape, batch, com_free, device):
        if self.prior_type == "gaussian" or self.prior_type == "normal":
            x0 = torch.randn(shape)
            if com_free:
                x0 = x0 - scatter_mean(x0, batch, dim=0)[batch]
        else:
            raise ValueError("Only Gaussian is supported")
        return x0.to(device)

    def step(self, xt, x_hat, batch, t_idx):
        """
        Perform a euler step in the continuous flow matching method.
        """
        if self.solver_type == "sde":
            data_scale, noise_scale, log_var = self.reverse_schedule(t_idx, batch)
            # data_scale = extract(self.posterior_mean_c0_coef, t, batch)
            # noise_scale = extract(self.posterior_mean_ct_coef, t, batch)
            # pos_log_variance = extract(self.posterior_logvar, t, batch)
            mean = data_scale * x_hat + noise_scale * xt
            # no noise when diffusion t == 0 so flow matching t == 1
            nonzero_mask = (1 - (t_idx == (self.timesteps - 1)).float())[batch].unsqueeze(-1)
            # ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
            x_next = mean + nonzero_mask * (0.5 * log_var).exp() * self.prior(
                xt.shape, batch, com_free=True, device=xt.device
            )
            # TODO: can add guidance module here
        else:
            raise ValueError("Only SDE Implemented")

        return x_next


class ContinuousFlowMatchingInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        schedule_params (dict): Type of interpolant schedule.
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        schedule_params: dict = {'type': 'linear', 'time': 'uniform'},
        prior_type: str = 'gaussian',
        update_weight_type: str = "constant",
        solver_type: str = "ode",
        timesteps: int = 500,
        min_t: float = 1e-2,
    ):
        super(ContinuousFlowMatchingInterpolant, self).__init__(schedule_params, prior_type, solver_type, timesteps)
        self.update_weight_type = update_weight_type
        self.min_t = min_t
        self.init_schedulers(schedule_params, timesteps)

    def init_schedulers(self, schedule_params, timesteps):
        self.schedule_type = schedule_params['type']
        if schedule_params['type'] == "linear":  #! vpe_linear is just linear with an update weight of recip_time_to_go
            self.discrete_time_only = False
            self.time = torch.linspace(self.min_t, 1, self.timesteps)
            self.forward_data_schedule = self.time  # lambda x: x/self.timesteps
            self.forward_noise_schedule = 1.0 - self.time  # lambda x: (1.0-x)/self.timesteps
        elif schedule_params['type'] == "vpe":
            # ! Doing this enforces discrete_time_only
            self.discrete_time_only = True
            self.alphas, self.alphas_prime = cosine_beta_schedule_fm(
                schedule_params, timesteps
            )  # FlowMol defines alpha as 1 - cos ^2
            self.forward_data_schedule = self.alphas
            self.reverse_data_schedule = 1.0 - self.alphas
            self.derivative_forward_data_schedule = self.alphas_prime
            self.alpha_bar = self.alphas  # For SNR

    def snr_loss_weight(self, t_idx=None, t=None):
        if t_idx is not None:
            if self.schedule_type == "linear":
                t = t_idx / self.timesteps
                return torch.clamp(t / (1 - t), min=0.05, max=1.5)
            else:
                return torch.clamp(self.snr(t_idx), min=0.05, max=1.5)
        else:
            return torch.clamp(t / (1 - t), min=0.05, max=1.5)

    def update_weight(self, t):
        if self.update_weight_type == "constant":
            weight = torch.ones_like(t).to(t.device)
        elif self.update_weight_type == "recip_time_to_go":
            weight = torch.clamp(1 / (1 - t), max=self.timesteps)  # at T = 1 this makes data_scale = 1
        return weight

    def forward_schedule(self, t=None, t_idx=None, batch=None):
        if t is not None and self.schedule_type == "linear":
            return t[batch].unsqueeze(1), (1.0 - t)[batch].unsqueeze(1)
        return (
            self.forward_data_schedule[t_idx].unsqueeze(1)[batch],
            self.forward_noise_schedule[t_idx].unsqueeze(1)[batch],
        )

    def reverse_schedule(self, batch, t_idx=None, t=None, t_next=None, dt=None):
        assert (
            (t is not None and dt is not None)
            or (t_idx is not None and dt is not None)
            or (t is not None and t_next is not None)
        ), "Must provide valid time."
        if dt is None:
            assert not self.discrete_time_only
            dt = (t_next - t)[batch]
        if self.schedule_params['type'] == "linear":
            if t is None:
                t = self.forward_data_schedule[t_idx]
            data_scale = self.update_weight(t[batch]) * dt
        elif self.schedule_params['type'] == "vpe":  # FlowMol
            data_scale = (self.derivative_forward_data_schedule[t_idx] * dt / (1 - self.forward_data_schedule[t_idx]))[
                batch
            ]  # alpha_prime[t]*dt/(1 - alpha[t]) #! EquiFm uses (1-a)^2 could be due to the definition of the scheduler FloMol uses cosine wheres EquiFm uses exp(- 0.5 * integral of betas(s)) where beta is some noise scheduler funtion

        return data_scale.unsqueeze(1), (1 - data_scale).unsqueeze(1)

    def interpolate(self, x1, batch, t=None, t_idx=None, com_free=True):
        """
        Interpolate using continuous flow matching method.
        """
        x0 = self.prior(x1.shape, batch, com_free, x1.device)
        data_scale, noise_scale = self.forward_schedule(t, t_idx, batch)
        return x1, data_scale * x1 + noise_scale * x0, x0

    def prior(self, shape, batch, com_free, device):
        if self.prior_type == "gaussian" or self.prior_type == "normal":
            x0 = torch.randn(shape)
            if com_free:
                x0 = x0 - scatter_mean(x0, batch, dim=0)[batch]
        else:
            raise ValueError("Only Gaussian is supported")
        return x0.to(device)

    def step(self, xt, x_hat, batch, t_idx=None, t=None, dt=None, t_next=None):
        """
        Perform a euler step in the continuous flow matching method.
        """
        # x_next = xt + self.update_weight(t) * dt * (x_hat - xt)
        data_scale, noise_scale = self.reverse_schedule(batch, t_idx, t, t_next, dt)
        x_next = data_scale * x_hat + noise_scale * xt
        return x_next


class DiscreteDiffusionInterpolant(Interpolant):
    """
    Class for continuous interpolation.
    Note the udnerlying D3PM only works for discrete time.
    Can look into MultiFlow Precurssor that uses CTMC for continuous time discrete diffusion (https://arxiv.org/pdf/2205.14987, https://github.com/andrew-cr/tauLDR/blob/main/lib/models/models.py)
    Argmax Flow also operate over discrete time
    Continuous time can work for cintuous gaussian representations found in DiffSBDD but we are not doing this.

    Attributes:
        schedule_params (dict): Type of interpolant schedule.
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        schedule_params: dict = {'type': 'd3pm'},
        prior_type: str = "uniform",
        solver_type: str = "sde",
        timesteps: int = 500,
        num_classes: int = 12,
    ):
        super(DiscreteDiffusionInterpolant, self).__init__(schedule_params, prior_type, solver_type, timesteps)
        self.num_classes = num_classes
        self.init_schedulers(schedule_params, timesteps)

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
        elif self.prior_type == "mask" or self.prior_type == "absorb":
            prior_dist = torch.zeros((self.num_classes))
            prior_dist[-1] = 1.0
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
        # import ipdb; ipdb.set_trace()
        return Qt, Qt_bar, Qt_bar_prev

    def init_schedulers(self, schedule_params, timesteps):
        self.schedule_type = schedule_params['type']
        self.alphas, self.betas = cosine_beta_schedule(schedule_params, timesteps, return_alpha=True)
        self.log_alpha = torch.log(self.alphas)
        self.log_alpha_bar = torch.cumsum(self.log_alpha, dim=0)
        self.alpha_bar = alphas_cumprod = torch.exp(self.log_alpha_bar)
        self.alpha_bar_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        if schedule_params['type'] == "d3pm":  # MIDI and EQ
            self.Qt, self.Qt_bar, self.Qt_prev_bar = self.d3pm_setup()
            self.forward_data_schedule = self.Qt_bar
            self.forward_noise_schedule = self.Qt_bar
        elif schedule_params['type'] == "argmax":  # TargetDiff
            self.forward_data_schedule = self.log_alpha_bar
            self.forward_noise_schedule = log_1_min_a(self.log_alpha_bar)

    def forward_schedule(self, t_idx, batch):
        # t = 1 - t
        t_idx = self.timesteps - 1 - t_idx
        return self.forward_data_schedule[t_idx][batch], self.forward_noise_schedule[t_idx][batch]

    def reverse_schedule(self, t_idx):
        t_idx = self.timesteps - 1 - t_idx
        return (
            self.reverse_data_schedule[t_idx].unsqueeze(1),
            self.reverse_noise_schedule[t_idx].unsqueeze(1),
            self.log_var[t_idx].unsqueeze(1),
        )

    def interpolate(self, x1, batch, t=None, t_idx=None):
        """
        Interpolate using discrete interpolation method.
        """
        # import ipdb; ipdb.set_trace()
        if len(x1.shape) == 1:
            x1_hot = F.one_hot(x1, self.num_classes)
        else:
            x1_hot = x1
        if self.schedule_type == "d3pm":
            assert self.discrete_time_only
            ford = self.forward_schedule(t_idx, batch)[0]
            probs = torch.einsum("nj, nji -> ni", [x1_hot.float(), ford])
            # probs = self.forward_schedule(t_idx, batch)[0] * x1  #! Eqn 3 of D3PM https://arxiv.org/pdf/2107.03006
            assert torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
            xt = probs.multinomial(
                1,
            ).squeeze()
            return x1, xt, probs
        elif self.schedule_type == "argmax":
            log_x0 = index_to_log_onehot(x1)
            data_scale, noise_scale = self.forward_schedule(t_idx)
            log_probs = log_add_exp(log_x0 + data_scale, noise_scale - np.log(self.num_classes))
            xt = log_sample_categorical(log_probs)

        return x1, xt, None

    def prior(self, shape, batch, device, one_hot=False):
        """
        Returns discrete index (num_samples,) or one hot if True (num_samples, num_classes)
        """
        num_samples = shape[0]
        if self.prior_type == "mask" or self.prior_type == "absorb":
            x0 = torch.ones((num_samples,)).to(torch.int64) * (self.num_classes - 1)
        elif self.prior_type == "uniform":
            x0 = torch.randint(0, self.num_classes, (num_samples,)).to(torch.int64)
        else:
            raise ValueError("Only uniform and mask are supported")
        if one_hot:
            x0 = F.one_hot(x0, num_classes=self.num_classes)
        return x0.to(device)

    def step(
        self,
        xt,
        x_hat,
        batch,
        t_idx,
    ):
        """
        Perform a euler step in the discrete interpolant method.
        """
        if self.solver_type == "sde" and self.schedule_type == "d3pm":
            assert self.discrete_time_only
            # TODO: Verify that this is correct
            # import ipdb; ipdb.set_trace()
            x_hat = F.one_hot(x_hat, num_classes=self.num_classes).float()
            # import ipdb; ipdb.set_trace(0)
            xt = F.one_hot(xt, num_classes=self.num_classes).float()
            t_idx = t_idx[batch]

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

            A = torch.einsum("nj, nji -> ni", [xt, self.Qt[t_idx].permute(0, 2, 1)]).unsqueeze(1)
            B = self.Qt_prev_bar[t_idx]
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


class DiscreteFlowMatchingInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        schedule_params (dict): Type of interpolant schedule.
        prior_type (str): Type of prior.
        update_weight_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        schedule_params: dict = {'type': 'linear'},
        prior_type: str = "uniform",
        update_weight_type: str = "d3pm",
        solver_type: str = "ode",
        timesteps: int = 500,
        num_classes: int = 10,
        min_t: float = 1e-2,
    ):
        super(DiscreteFlowMatchingInterpolant, self).__init__(schedule_params, prior_type, solver_type, timesteps)
        self.num_classes = num_classes
        self.update_weight_type = update_weight_type
        self.min_t = min_t
        self.init_schedulers(schedule_params, timesteps)

    def init_schedulers(self, schedule_params, timesteps):
        self.schedule_type = schedule_params['type']
        if schedule_params['type'] == "linear":  #! vpe_linear is just linear with an update weight of recip_time_to_go
            self.discrete_time_only = False
            time = torch.linspace(self.min_t, 1, self.timesteps)
            self.forward_data_schedule = time
            self.forward_noise_schedule = 1.0 - time
        elif schedule_params['type'] == "vpe":
            self.discrete_time_only = True
            self.alphas, self.alphas_prime = cosine_beta_schedule_fm(
                schedule_params, timesteps
            )  # FlowMol defines alpha as 1 - cos ^2
            self.forward_data_schedule = self.alphas
            self.reverse_data_schedule = 1.0 - self.alphas
            self.derivative_forward_data_schedule = self.alphas_prime
            self.alpha_bar = self.alphas

    def snr_loss_weight(self, t_idx=None, t=None):
        if t_idx is not None:
            if self.schedule_type == "linear":
                t = t_idx / self.timesteps
                return torch.clamp(t / (1 - t), min=0.05, max=1.5)
            else:
                return torch.clamp(self.snr(t_idx), min=0.05, max=1.5)
        else:
            return torch.clamp(t / (1 - t), min=0.05, max=1.5)

    def update_weight(self, t):
        if self.update_weight_type == "constant":
            weight = torch.ones_like(t).to(t.device)
        elif self.update_weight_type == "recip_time_to_go":
            weight = torch.clamp(1 / (1 - t), max=self.timesteps)  # at T = 1 this makes data_scale = 1
        return weight

    def forward_schedule(self, t=None, t_idx=None, batch=None):
        if t is not None and self.schedule_type == "linear":
            return t[batch].unsqueeze(1), (1.0 - t)[batch].unsqueeze(1)
        return (
            self.forward_data_schedule[t_idx].unsqueeze(1)[batch],
            self.forward_noise_schedule[t_idx].unsqueeze(1)[batch],
        )

    def reverse_schedule(self, batch, t_idx=None, t=None, t_next=None, dt=None):
        assert (
            (t is not None and dt is not None)
            or (t_idx is not None and dt is not None)
            or (t is not None and t_next is not None)
        ), "Must provide valid time."
        if dt is None:
            assert not self.discrete_time_only
            dt = (t_next - t)[batch]
        if self.schedule_params['type'] == "linear":
            if t is None:
                t = self.forward_data_schedule[t_idx]
            data_scale = self.update_weight(t[batch]) * dt
        elif self.schedule_params['type'] == "vpe":  # FlowMol
            data_scale = (self.derivative_forward_data_schedule[t_idx] * dt / (1 - self.forward_data_schedule[t_idx]))[
                batch
            ]  # alpha_prime[t]*dt/(1 - alpha[t]) #! EquiFm uses (1-a)^2 could be due to the definition of the scheduler FloMol uses cosine wheres EquiFm uses exp(- 0.5 * integral of betas(s)) where beta is some noise scheduler funtion

        return data_scale.unsqueeze(1), (1 - data_scale).unsqueeze(1)

    def interpolate(self, x1, batch, t=None, t_idx=None):
        """
        Interpolate using discrete interpolation method.
        """
        if self.prior_type in ["mask", "absorb", "uniform"]:
            x0 = self.prior(x1.shape, self.num_classes, x1.device).unsqueeze(1)
            if t is None:
                t = t_idx / self.timesteps
            t = t[batch]
            xt = x1.clone().unsqueeze(1)
            corrupt_mask = torch.rand((x1.shape[0], 1)).to(t.device) < (1 - t.unsqueeze(1))  # [:, None])
            xt[corrupt_mask] = x0[corrupt_mask]
        else:
            raise ValueError("Only uniform and mask are supported")

        return x1, xt.squeeze(1), x0.squeeze(1)

    def prior(self, shape, batch, device, one_hot=False):
        """
        Returns discrete index (num_samples, 1) or one hot if True (num_samples, num_classes)
        """
        num_samples = shape[0]
        if self.prior_type == "mask" or self.prior_type == "absorb":
            x0 = torch.ones((num_samples,)).to(torch.int64) * (self.num_classes - 1)
        elif self.prior_type == "uniform":
            x0 = torch.randint(0, self.num_classes, (num_samples,)).to(torch.int64)
        else:
            raise ValueError("Only uniform and mask/absorb are supported")
        if one_hot:
            x0 = F.one_hot(x0, num_classes=self.num_classes)
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
        stochasticity=1,
        temp=0.1,
        use_purity=False,
        last_step=False,
    ):
        """
        Perform a euler step in the discrete interpolant method.
        """

        N = stochasticity
        S = self.num_classes
        MASK_TOKEN_INDEX = S - 1
        if t is None:
            t = t_idx / self.timesteps
        if dt is None:
            assert not self.discrete_time_only
            dt = t_next - t
            dt = dt[batch].unsqueeze(1)
        t = t[batch].unsqueeze(1)
        if self.prior_type == "uniform":
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


def test_continuous_diffusion(ligand_pos, batch_ligand):
    print("ContinuousDiffusionInterpolant")
    pos_interpolant = ContinuousDiffusionInterpolant()
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='continuous diffusion interpolation', total=len(time_seq)):
        t_idx = pos_interpolant.sample_time_idx(4, method='stab_mode')
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

    for i in tqdm(time_seq, desc='continuous time_idx interpolation', total=len(time_seq)):
        t_idx = pos_interpolant.sample_time_idx(4, method='uniform')
        data_scale, noise_scale = pos_interpolant.forward_schedule(t_idx=t_idx, batch=batch_ligand)
        x1, xt, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t_idx=t_idx, com_free=True)

    xt = pos_interpolant.prior(x1.shape, batch_ligand, com_free=True, device=x1.device)
    dt = 1 / 500
    check = [torch.sum((ligand_pos - xt) ** 2)]
    for i in tqdm(time_seq, desc='continuous time_index flow step static dt', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t_idx=t_idx, com_free=True)
        x_hat = x1 + 0.0001 * pos_interpolant.prior(x1.shape, batch_ligand, True, x1.device)
        # if i == 499:
        #     import ipdb; ipdb.set_trace()
        x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t_idx=t_idx, dt=dt)
        xt = x_tp1
        check.append(torch.sum((ligand_pos - xt) ** 2))
    # import ipdb; ipdb.set_trace()
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
    # import ipdb; ipdb.set_trace()
    #! linspace is inclusing just raw discrete time*timeesteps is not this is a discrepancy that is not noticable in diffusion when everything is indexed.
    # ! We are going to follow MultiFlow for this so we add a clamp to the forward schedule for the t = 1 pass
    xt = pos_interpolant.prior(x1.shape, batch_ligand, com_free=True, device=x1.device)
    check = [torch.sum((ligand_pos - xt) ** 2)]
    time_seq = torch.linspace(1e-2, 1, 500)  # min_t used in multi flow
    for i in tqdm(range(1, len(time_seq)), desc='continuous time flow step dynamic dt', total=len(time_seq)):
        t = torch.full(size=(4,), fill_value=time_seq[i - 1], device='cpu')
        t_next = torch.full(size=(4,), fill_value=time_seq[i], device='cpu')
        # import ipdb; ipdb.set_trace()
        x1, xt01, x0 = pos_interpolant.interpolate(ligand_pos, batch_ligand, t=t, com_free=True)
        x_hat = x1 + 0.0001 * pos_interpolant.prior(x1.shape, batch_ligand, True, x1.device)
        x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t=t, t_next=t_next)
        xt = x_tp1
        check.append(torch.sum((ligand_pos - xt) ** 2))
    x_tp1 = pos_interpolant.step(xt, x_hat, batch_ligand, t=t, t_next=t_next)
    # import ipdb; ipdb.set_trace()


def test_discrete_diffusion(h, batch):
    print("DiscreteDiffusionInterpolant-Uniform: ONLY WORKS with DISCRETE TIME")
    num_classes = 13

    interpolant = DiscreteDiffusionInterpolant(num_classes=num_classes)
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete diffusion interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time_idx(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch_ligand)
        x1, xt, probs = interpolant.interpolate(h, batch, t_idx=t_idx)
    assert all(torch.argmax(probs, 1) == x1)

    xt = interpolant.prior(h.shape, batch, device=h.device)
    for i in tqdm(time_seq, desc='discrete diffusion step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, probs = interpolant.interpolate(h, batch_ligand, t_idx=t_idx)
        x_hat = xt01
        x_tp1 = interpolant.step(xt, x_hat, batch, t_idx)
        xt = x_tp1

    print("DiscreteDiffusionInterpolant-Absorb")
    num_classes = 14

    interpolant = DiscreteDiffusionInterpolant(num_classes=num_classes, prior_type="absorb")
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete diffusion absorb interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time_idx(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch_ligand)
        x1, xt, probs = interpolant.interpolate(h, batch, t_idx=t_idx)
    probs[:, -1] = 0
    assert all(torch.argmax(probs, 1) == x1)

    xt = interpolant.prior(h.shape, batch, device=h.device)
    for i in tqdm(time_seq, desc='discrete diffusion absorb step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, probs = interpolant.interpolate(h, batch_ligand, t_idx=t_idx)
        x_hat = xt01
        x_tp1 = interpolant.step(xt, x_hat, batch, t_idx)
        xt = x_tp1


def test_discrete_flowmatching(h, batch):
    print("DiscreteDiffusionInterpolant-Uniform: ONLY WORKS with DISCRETE TIME")
    num_classes = 13

    interpolant = DiscreteFlowMatchingInterpolant(num_classes=num_classes, prior_type="uniform")
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete flowmatching interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time_idx(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch_ligand)
        x1, xt, x0 = interpolant.interpolate(h, batch, t_idx=t_idx)

    xt = interpolant.prior(h.shape, batch, device=h.device)
    for i in tqdm(time_seq, desc='discrete diffusion step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, x0 = interpolant.interpolate(h, batch_ligand, t_idx=t_idx)
        x_hat = F.one_hot(xt01, num_classes)
        x_tp1 = interpolant.step(xt, x_hat, batch, t_idx=t_idx, dt=1 / 500)
        xt = x_tp1

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
    for i in tqdm(range(1, len(time_seq) + 1), desc='continuous time flow step dynamic dt', total=len(time_seq)):
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

    interpolant = DiscreteFlowMatchingInterpolant(num_classes=num_classes, prior_type="absorb")
    time_seq = list(range(0, 500))
    for i in tqdm(time_seq, desc='discrete diffusion absorb interpolation', total=len(time_seq)):
        t_idx = interpolant.sample_time_idx(4, method='stab_mode')
        data_scale, _ = interpolant.forward_schedule(t_idx, batch_ligand)
        x1, xt, x0 = interpolant.interpolate(h, batch, t_idx=t_idx)

    xt = interpolant.prior(h.shape, batch, device=h.device)
    for i in tqdm(time_seq, desc='discrete diffusion absorb step', total=len(time_seq)):
        t_idx = torch.full(size=(4,), fill_value=i, dtype=torch.int64, device='cpu')
        x1, xt01, x0 = interpolant.interpolate(h, batch_ligand, t_idx=t_idx)
        x_hat = F.one_hot(xt01, num_classes)
        x_tp1 = interpolant.step(xt, x_hat, batch, t_idx=t_idx, dt=1 / 500)
        xt = x_tp1

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
    for i in tqdm(range(1, len(time_seq) + 1), desc='continuous time flow step dynamic dt', total=len(time_seq)):
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


if __name__ == "__main__":
    from tqdm import tqdm

    print("TEST")
    ligand_pos = torch.rand((75, 3))
    batch_ligand = torch.Tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]
    ).to(torch.int64)
    ligand_feats = torch.Tensor(
        [
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            1,
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            12,
            2,
            5,
            2,
            3,
            5,
            1,
            5,
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            1,
        ]
    ).to(torch.int64)

    test_continuous_diffusion(ligand_pos, batch_ligand)
    test_continuous_flowmatching(ligand_pos, batch_ligand)
    test_discrete_diffusion(ligand_feats, batch_ligand)
    test_discrete_flowmatching(ligand_feats, batch_ligand)
    print("SUCCESS")
