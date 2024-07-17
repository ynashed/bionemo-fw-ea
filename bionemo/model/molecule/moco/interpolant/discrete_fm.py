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
import torch.nn.functional as F
from torch_geometric.utils import sort_edge_index

from bionemo.model.molecule.moco.interpolant.interpolant import Interpolant
from bionemo.model.molecule.moco.interpolant.interpolant_scheduler import build_scheduler


class DiscreteFlowMatchingInterpolant(Interpolant):
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
        # schedule_params: dict = {'type': 'linear', 'time': 'uniform', 'time_type': 'continuous'},
        prior_type: str = "uniform",
        vector_field_type: str = "standard",
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
        self.vector_field_type = vector_field_type
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
            self.discrete_time_only = True
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
        #! No loss weightining is used for discrete data in MultiFlow or Semla
        weight = torch.ones_like(time).to(time.device)
        return weight
        # if self.time_type == "continuous":
        #     # return torch.clamp(time / (1 - time), min=0.05, max=1.5)
        #     # loss scale for "frameflow":
        #     return (1 - torch.clamp(t, 0, self.max_t_sample)) / (1 - torch.clamp(t, 0, self.clip_t))**2
        # else:
        #     if self.schedule_type == "linear":
        #         t = time / self.timesteps
        #         return (1 - torch.clamp(t, 0, self.max_t_sample)) / (1 - torch.clamp(t, 0, self.clip_t))**2
        #     else:
        #         return torch.clamp(self.snr(time), min=0.05, max=1.5)

    def update_weight(self, t):
        # if self.vector_field_type == "endpoint":
        weight = torch.ones_like(t).to(t.device)
        # elif self.vector_field_type == "standard":
        # weight = torch.clamp(1 / (1 - t), max=self.timesteps)  # at T = 1 this makes data_scale = 1
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

    # def reverse_schedule(self, batch, time, dt):
    #     if self.time_type == "continuous":
    #         if self.schedule_type == "linear":
    #             data_scale = self.update_weight(time[batch]) * dt
    #     else:
    #         if self.schedule_type == "linear":
    #             t = self.forward_data_schedule[time]
    #             data_scale = self.update_weight(t[batch]) * dt
    #         elif self.schedule_type == "vpe":  # FlowMol
    #             data_scale = (
    #                 self.derivative_forward_data_schedule[time] * dt / (1 - self.forward_data_schedule[time])
    #             )[
    #                 batch
    #             ]  # alpha_prime[t]*dt/(1 - alpha[t]) #! EquiFm uses (1-a)^2 could be due to the definition of the scheduler FloMol uses cosine wheres EquiFm uses exp(- 0.5 * integral of betas(s)) where beta is some noise scheduler funtion

    #     return data_scale.unsqueeze(1), (1 - data_scale).unsqueeze(1)

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
