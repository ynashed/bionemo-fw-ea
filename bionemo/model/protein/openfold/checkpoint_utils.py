from pathlib import Path
from typing import Union

import torch
from nemo.utils import logging

from bionemo.model.protein.openfold.openfold_model import AlphaFold


def remap_layers_and_load(
    alphafold: AlphaFold,
    checkpoint_filepath: Path,
) -> None:
    openfold_params_07_27 = torch.load(checkpoint_filepath)
    alphafold_state_dict_keys = set(alphafold.state_dict().keys())
    init_state_dict = {}
    for key, tensor in openfold_params_07_27.items():
        # these layer names are converted to fit public OpenFold checkpoint as available on 07/27/2023
        key = key.replace("template_pointwise_att.", "template_pointwise_attention.")
        key = key.replace("evoformer.", "evoformer_stack.")
        key = key.replace("aux_heads.", "auxiliary_heads.")
        key = key.replace("._msa_att.", ".")
        key = key.replace(".transition.layers.0.", ".transition.")
        assert key in alphafold_state_dict_keys
        assert isinstance(tensor, torch.Tensor)
        init_state_dict[key] = tensor
    alphafold.load_state_dict(init_state_dict, strict=True)


def load_pt_checkpoint(model: AlphaFold, checkpoint_path: Union[str, Path]):
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        return
    except RuntimeError:
        logging.warning(
            f"Failed to directly load {checkpoint_path} checkpoint. Trying again with layer names remapping."
        )

    # do not recover if mapping fails
    remap_layers_and_load(model, checkpoint_path)
