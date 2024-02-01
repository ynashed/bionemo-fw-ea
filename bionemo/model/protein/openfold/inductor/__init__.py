import torch

from bionemo.data.protein.openfold.helpers import is_ampere_arch, is_hopper_arch


_ENABLED = False


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def disable():
    global _ENABLED
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def is_enabled_and_autograd_off() -> bool:
    return _ENABLED and not torch.is_grad_enabled()


def is_enabled_on_hopper() -> bool:
    return _ENABLED and is_hopper_arch()


def is_enabled_on_hopper_and_autograd_off() -> bool:
    return _ENABLED and is_hopper_arch() and not torch.is_grad_enabled()


def is_enabled_on_ampere() -> bool:
    return _ENABLED and is_ampere_arch()


def is_enabled_on_ampere_and_autograd_off() -> bool:
    return _ENABLED and is_ampere_arch() and not torch.is_grad_enabled()


def is_enabled_on_ampere_and_autograd_on() -> bool:
    return _ENABLED and is_ampere_arch() and torch.is_grad_enabled()
