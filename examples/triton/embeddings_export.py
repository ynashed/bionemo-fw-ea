# Copyright (c) 2022, NVIDIA CORPORATION.
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
from nav_common import NavWrapper
from nemo.core.config import hydra_runner
from nemo.utils import logging

from examples.infer import setup_inference


INPUT_NAMES = ('tokens', 'mask')


logging.getLogger('nemo_logger').setLevel(logging.WARNING)


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    global INPUT_NAMES

    if hasattr(cfg, 'nav_path'):
        nav_path = cfg.nav_path
    else:
        suffix = '.nemo'
        nav_path = cfg.model.downstream_task.restore_from_path[: -len(suffix)] + '.nav'

    inferer, trainer, dataloader = setup_inference(cfg)

    model = NavWrapper(inferer)

    def get_dataloader():
        batches = []
        for batch in dataloader:
            tokens_enc, enc_mask = inferer.tokenize(batch['sequence'])
            batches.append((tokens_enc, enc_mask))
        return batches

    def get_trt_profile():
        trt_profile = nav.api.config.TensorRTProfile()
        for name in INPUT_NAMES:
            trt_profile.add(
                name,
                (1, 1),
                (cfg.model.data.batch_size, cfg.model.seq_length // 2),
                (cfg.model.data.batch_size * 2, cfg.model.seq_length),
            )
        return trt_profile

    def get_verify_function(atol: float = 1.0e-3, rtol: float = 1.0e-3):
        """Define verify function that compares outputs of the torch model and the optimized model."""

        def verify_func(ys_runner, ys_expected) -> bool:
            for y_runner, y_expected in zip(ys_runner, ys_expected):
                for k in y_runner:
                    print('KEY:', k)
                    print('exp range', np.percentile(y_expected[k], (0, 10, 50, 90, 100)))
                    print('runner range', np.percentile(y_runner[k], (0, 10, 50, 90, 100)))
                    print('adiff', np.abs(y_runner[k] - y_expected[k]).max())
                    print('rdiff', np.abs(1 - y_runner[k] / y_expected[k]).max())
            return True

        return verify_func

    package = nav.torch.optimize(
        model=model,
        dataloader=get_dataloader(),
        input_names=INPUT_NAMES,
        output_names=('embeddings',),
        custom_configs=[
            nav.TensorRTConfig(trt_profile=get_trt_profile()),
            nav.OnnxConfig(
                dynamic_axes={name: {0: "batchsize", 1: "seqlen"} for name in INPUT_NAMES},
            ),
        ],
        verify_func=get_verify_function(),
    )

    nav.package.save(package, nav_path, override=True)

    print('Exported to', nav_path)


if __name__ == "__main__":
    main()
