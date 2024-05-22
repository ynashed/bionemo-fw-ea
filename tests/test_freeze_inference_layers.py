import pytest
from omegaconf import open_dict

from bionemo.model.protein.esm1nv import ESM1nvInference
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


@pytest.fixture(scope="module")
def config_path(bionemo_home) -> str:
    path = bionemo_home / "examples" / "protein" / "esm1nv" / "conf"
    return str(path.absolute())


@pytest.fixture(scope="module")
def inference_model_3_frozen(config_path) -> ESM1nvInference:
    cfg = load_model_config(config_name="infer", config_path=config_path)
    with open_dict(cfg):
        cfg.model['freeze_layers'] = 3
    # load score model
    initialize_distributed_parallel_state()
    model = ESM1nvInference(cfg)
    model.eval()
    yield model
    teardown_apex_megatron_cuda()


def test_partially_frozen_model_3_layers(inference_model_3_frozen):
    # check that the first 3 layers are frozen, and the last 3 layers are trainable
    for i in range(6):
        for param in inference_model_3_frozen.model.model.language_model.encoder.layers[i].parameters():
            if i < 3:
                assert param.requires_grad is False
            else:
                assert param.requires_grad is True
