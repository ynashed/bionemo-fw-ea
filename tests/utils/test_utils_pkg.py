import torch

from bionemo.utils import lookup_or_use


def test_lookup_or_use() -> None:
    m = lookup_or_use(torch.nn, "Module")
    assert isinstance(m, torch.nn.Module)
