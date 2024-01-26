# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Entry point to DiffDock inference: generating ligand poses.

modify parameters from conf/*.yaml
"""
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch_geometric.seed import seed_everything

from bionemo.data.diffdock.inference import build_inference_datasets
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference, do_inference_sampling


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    seed = cfg.get('seed', None)
    if seed is not None:
        seed_everything(seed)

    # process input and build inference datasets for score model, and confidence model, build dataloader
    complex_name_list, test_dataset, confidence_test_dataset, test_loader = build_inference_datasets(cfg)

    # load score model
    model = DiffDockModelInference(cfg.score_infer)

    # load confidence model
    if 'confidence_infer' in cfg:
        confidence_model = DiffDockModelInference(cfg.confidence_infer)
    else:
        confidence_model = None

    # Perform the inference, which as following steps
    # 1. randomize the initial ligand positions.
    # 2. Doing reverse diffusion sampling with the score model and get confidence scores for the generated ligand poses.
    # 3. Write out results.
    failures, skipped = do_inference_sampling(
        cfg, model, confidence_model, complex_name_list, test_loader, test_dataset, confidence_test_dataset
    )

    logging.info(f'Failed for {failures} complexes')
    logging.info(f'Skipped {skipped} complexes')
    logging.info(f'Results are in {cfg.out_dir}')


if __name__ == '__main__':
    main()
