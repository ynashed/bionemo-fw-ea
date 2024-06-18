# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


def build_scheduler(
    scheduler_type: str, num_diffusion_timesteps: int, s: float = 0.008, sqrt: bool = False, nu: float = 1.0, clip=True
):
    """
    Factory function to build a scheduler based on the provided parameters.

    Args:
        scheduler_params (dict): Parameters for building the scheduler, must include "schedule_type".

    Returns:
        InterpolantDiffusionScheduler: An instance of a scheduler.

    Raises:
        NotImplementedError: If the scheduler type is not implemented.
    """
    if scheduler_type == "cosine_adaptive":
        return CosineSchedule(num_diffusion_timesteps, s, sqrt, nu, clip)
    elif scheduler_type == "linear":
        return LinearSchedule(num_diffusion_timesteps)
    else:
        raise NotImplementedError(f"Scheduler '{scheduler_type}' is not implemented")


class InterpolantDiffusionSchedule(nn.Module, ABC):
    """
    Abstract base class for diffusion schedulers that compute alpha and beta values for diffusion processes.

    Methods:
        get_alphas_and_betas() -> Tuple[torch.Tensor, torch.Tensor]:
            Returns the alpha and beta values for the diffusion process.

        compute_alphas() -> torch.Tensor:
            Abstract method to compute the alpha values. Subclasses must implement this method.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('alphas', self.compute_alphas())
        log_alphas = torch.log(self.alphas)
        log_alphas_bar = torch.cumsum(log_alphas, dim=0)
        self.register_buffer('alphas_bar', torch.exp(log_alphas_bar))

    def get_alphas_and_betas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the alpha and beta values for the diffusion process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Alpha and beta values.
        """
        return self.alphas, 1 - self.alphas

    def get_alphas_bar(self):
        return self.alphas_bar

    @abstractmethod
    def compute_alphas(self) -> torch.Tensor:
        """
        Abstract method to compute the alpha values. Subclasses must implement this method.

        Returns:
            torch.Tensor: Alpha values.
        """
        pass


class LinearSchedule(InterpolantDiffusionSchedule):
    """
    Linear scheduler for diffusion processes. This class computes the alpha values linearly.

    Args:
        num_diffusion_timesteps (int): Number of diffusion time steps.
    """

    def __init__(self, num_diffusion_timesteps: int, **kwargs):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        super().__init__()

    def compute_alphas(self) -> torch.Tensor:
        """
        Computes the alpha values linearly.

        Returns:
            torch.Tensor: Alpha values.
        """
        alphas = torch.linspace(1, 0, self.num_diffusion_timesteps + 1)[:-1]
        return alphas.clip(min=0.001, max=1.0)


class CosineSchedule(InterpolantDiffusionSchedule):
    """
    Cosine scheduler for diffusion processes. This class computes the alpha values using a cosine function.

    Args:
        num_diffusion_timesteps (int): Number of diffusion time steps.
        s (float): Smoothing parameter for the cosine function. Default is 0.008.
        sqrt (bool): Whether to take the square root of the alpha values. Default is False.
        nu (float): Exponent for the cosine function. Default is 1.0.
        **kwargs: Additional arguments.
    """

    def __init__(
        self, num_diffusion_timesteps: int, s: float = 0.008, sqrt: bool = False, nu: float = 1.0, clip=True, **kwargs
    ):
        self.s = s
        self.nu = nu
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.sqrt = sqrt
        self.clip = clip
        super().__init__()

    def clip_noise_schedule(self, alphas2, clip_value=0.01) -> torch.Tensor:
        """
        For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
        sampling.
        """
        alphas2 = torch.concat([torch.ones(1), alphas2], dim=0)
        alphas_step = alphas2[1:] / alphas2[:-1]
        alphas_step = alphas_step.clip(min=clip_value, max=1.0)
        alphas2 = torch.cumprod(alphas_step, dim=0)
        return alphas2

    def compute_alphas(self) -> torch.Tensor:
        """
        Computes the alpha values using a cosine function.

        Returns:
            torch.Tensor: Alpha values.
        """
        steps = self.num_diffusion_timesteps + 2
        x = torch.linspace(0, self.num_diffusion_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.num_diffusion_timesteps) ** self.nu + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
        )
        if self.clip:
            alphas_cumprod = self.clip_noise_schedule(alphas_cumprod, clip_value=0.05)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        if self.sqrt:
            alphas = torch.sqrt(alphas)
        alphas = alphas.clip(min=0.001, max=1.0)
        return alphas[1:]  # Removing the extra piece


def test_scheduler():
    """
    Test function to verify the scheduler implementation.

    Asserts that the length of the computed alpha values matches the expected number of diffusion time steps.
    Prints the computed alpha values.
    """
    scheduler_params = {
        "schedule_type": "cosine_adaptive",
        "num_diffusion_timesteps": 10,
        "nu": 1.5,
        "s": 0.008,
        "sqrt": False,
    }
    scheduler = build_scheduler(scheduler_params=scheduler_params)
    assert len(scheduler.alphas) == scheduler_params["num_diffusion_timesteps"]

    alphas, betas = scheduler.get_alphas_and_betas()
    alphas_bar = scheduler.get_alphas_bar()
    assert torch.abs(alphas[0] * alphas[1] - alphas_bar[1]) < 1e-7
    print("Cosine Scheduler Alphas:", alphas)

    scheduler_params["schedule_type"] = "linear"
    scheduler = build_scheduler(scheduler_params=scheduler_params)
    assert len(scheduler.alphas) == scheduler_params["num_diffusion_timesteps"]

    alphas, betas = scheduler.get_alphas_and_betas()
    print("Linear Scheduler Alphas:", alphas)


if __name__ == "__main__":
    test_scheduler()
