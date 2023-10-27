# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import model_navigator as nav
import numpy as np
import torch
from nav_common import NavWrapper
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf.omegaconf import OmegaConf
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from bionemo.model.utils import initialize_distributed_parallel_state


def get_mask_postprocess_fn(cfg):
    # FIXME this is bug prone!!!
    if cfg.target == "bionemo.model.protein.esm1nv.esm1nv_model.ESM1nvModel":
        print('ESM postprocess')

        def fun(enc_mask: torch.Tensor) -> torch.Tensor:
            enc_mask[:, 0:2] = 0
            enc_mask = torch.roll(enc_mask, shifts=-1, dims=1)
            return enc_mask

    else:

        def fun(enc_mask: torch.Tensor) -> torch.Tensor:
            return enc_mask

    return fun


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if hasattr(cfg, 'nav_path'):
        nav_path = cfg.nav_path
    else:
        suffix = '.nemo'
        nav_path = cfg.model.downstream_task.restore_from_path[: -len(suffix)] + '.nav'

    infer_class = import_class_by_path(cfg.infer_target)

    initialize_distributed_parallel_state(
        local_rank=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_split_rank=0,
    )

    MODEL = infer_class(cfg)
    MODEL.freeze()
    nav_model = NavWrapper(MODEL)

    mask_postprocess_fn = get_mask_postprocess_fn(cfg)

    package = nav.package.load(nav_path)
    package.load_source_model(nav_model)
    runner = package.get_runner(strategy=nav.MaxThroughputStrategy())

    runner.activate()

    @batch
    def _infer_fn(sequences: np.ndarray):
        sequences = np.char.decode(sequences.astype("bytes"), "utf-8")
        sequences = sequences.squeeze(1).tolist()

        tokens_enc, enc_mask = MODEL.tokenize(sequences)
        inp = {
            'tokens': tokens_enc.cpu().detach().numpy(),
            'mask': enc_mask.cpu().detach().numpy(),
        }

        hidden_states = runner.infer(inp)
        hidden_states = torch.tensor(hidden_states['embeddings'], device='cuda')
        enc_mask = mask_postprocess_fn(enc_mask)

        embedding = MODEL.hiddens_to_embedding(hidden_states, enc_mask)

        response = {
            "embedding": embedding.cpu().numpy(),
        }

        return response

    with Triton() as triton:
        logging.info("Loading model")
        triton.bind(
            model_name="bionemo_model",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="sequences", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="embedding", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=10),
        )
        logging.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    main()
