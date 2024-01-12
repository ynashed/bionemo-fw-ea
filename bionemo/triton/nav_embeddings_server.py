# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Sequence

import model_navigator
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from pytriton.triton import Triton

from bionemo.triton.embeddings import nav_triton_embedding_infer_fn
from bionemo.triton.types_constants import BIONEMO_MODEL, EMBEDDINGS, SEQUENCES
from bionemo.triton.utils import load_navigated_model_for_inference, register_str_embedding_infer_fn


__all_: Sequence[str] = ()


@hydra_runner(config_path="conf", config_name="infer")
def entrypoint(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    model, runner = load_navigated_model_for_inference(cfg, strategy=model_navigator.MaxThroughputStrategy())

    with Triton() as triton:
        register_str_embedding_infer_fn(
            triton,
            nav_triton_embedding_infer_fn(model, runner),
            triton_model_name=BIONEMO_MODEL,
            in_name=SEQUENCES,
            out=EMBEDDINGS,
            max_batch_size=10,
            verbose=True,
        )
        logging.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    entrypoint()
