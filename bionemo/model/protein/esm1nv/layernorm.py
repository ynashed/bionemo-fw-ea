import torch
from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm
from nemo.utils import logging


def esm_get_layer_norm(normalized_shape, *args, **kwargs):
    # TODO(srabhi, georgea): refactor the custom esm_get_layer_norm module using Megatron Core when NeMo 1.21 is available
    use_pt_layernorm = kwargs.pop('use_pt_layernorm', False)
    if use_pt_layernorm:
        logging.warning("Using PyTorch LayerNorm instead of the default NeMo version")
        eps = kwargs.pop('eps', 1e-05)
        return torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True)
    else:
        return get_layer_norm(normalized_shape, *args, **kwargs)
