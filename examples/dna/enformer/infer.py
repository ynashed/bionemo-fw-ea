# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bionemo.data.dna.enformer.genome_interval import FastaDataset
from bionemo.model.dna.enformer import Enformer
from bionemo.model.dna.enformer.writer import FastaRecordsWriter
from bionemo.model.utils import setup_trainer


@hydra_runner(config_path="conf", config_name="enformer_infer")
def main(cfg):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    writer_callback = FastaRecordsWriter(
        output_dir=cfg.model.predictions_output_path, mode=cfg.model.predictions_write_mode
    )
    trainer = setup_trainer(cfg, callbacks=[writer_callback])
    enformer = Enformer.restore_from(restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer)
    ds = FastaDataset(fasta_file=cfg.infer_path, context_length=cfg.model.context_length)
    dl = DataLoader(ds, batch_size=cfg.model.micro_batch_size, num_workers=cfg.model.data.num_workers)
    trainer.predict(enformer, dl, return_predictions=False)


if __name__ == '__main__':
    main()
