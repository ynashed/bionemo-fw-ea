# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.data.protein.openfold.datahub import get_predict_dataloader
from bionemo.model.protein.openfold.checkpoint_utils import load_pt_checkpoint
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.writer import PredictionFeatureWriter, PredictionPDBWriter
from bionemo.model.utils import setup_trainer


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    cfg = instantiate(cfg)
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing required keys in config:\n{missing_keys}")

    np.random.seed(cfg.model.seed)
    pl.seed_everything(cfg.model.seed)

    writers = [PredictionPDBWriter(cfg.results_path, cfg.force)]
    if cfg.model.downstream_task.outputs:
        writers.append(PredictionFeatureWriter(cfg.results_path, cfg.model.downstream_task.outputs, cfg.force))
    trainer = setup_trainer(cfg, callbacks=writers)

    if cfg.get('restore_from_path', None):
        alphafold = AlphaFold.restore_from(
            restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer
        )
    elif cfg.get('torch_restore', None):
        alphafold = AlphaFold(cfg=cfg.model, trainer=trainer)
        load_pt_checkpoint(model=alphafold, checkpoint_path=cfg.torch_restore)
    else:
        raise ValueError(
            'No checkpoint has been provided neither via restore_from_path nor torch_restore. \
                           Inference was not ran.'
        )

    predict_dataloader = get_predict_dataloader(model_cfg=cfg.model, predict_session_cfg=cfg)
    trainer.predict(alphafold, predict_dataloader, return_predictions=False)


if __name__ == '__main__':
    main()
