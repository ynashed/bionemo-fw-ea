# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#!/usr/bin/env python
from pathlib import Path

from nemo import lightning as nl
from nemo.utils import logging

from bionemo.contrib.data.singlecell.datamodule import SingleCellDataModule
from bionemo.contrib.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.contrib.model.singlecell.geneformer import GeneformerConfig, GeneformerModel


# Assuming you have data mounted to your base bionemo dir... Change this for release when bionemo2 becomes base.
# This is the full data path:
# pretrain_data_path =  Path(__file__).parent.parent.parent.parent.parent / "data/cellxgene_2023-12-15/processed_data"
# Test datapath:
# aws s3 cp s3://general-purpose/cellxgene_2023-12-15_small /workspace/bionemo/data/cellxgene_2023-12-15_small --recursive --endpoint-url https://pbss.s8k.io
# then this script will work from any directory
# python path/to/pretrain.py
pretrain_data_path = (
    Path(__file__).parent.parent.parent.parent.parent / "data/cellxgene_2023-12-15_small/processed_data"
)
train_data_path = pretrain_data_path / "train"
val_data_path = pretrain_data_path / "val"
test_data_path = pretrain_data_path / "test"


if __name__ == "__main__":
    devices, seq_length = 1, 2048

    strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
    trainer = nl.Trainer(
        devices=devices,
        max_steps=5,
        accelerator="gpu",
        strategy=strategy,
        # precision=None,z
        # plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=True),
    )

    # data = llm.PreTrainingDataModule(
    #     data_path, seq_length=seq_length, global_batch_size=8, micro_batch_size=2
    # )
    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {'tokenizer': tokenizer, 'median_dict': median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")
    mbs = 8
    # gpus = 1
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=train_data_path,
        val_dataset_path=val_data_path,
        test_dataset_path=test_data_path,
        median_dict=median_dict,
        micro_batch_size=mbs,
        global_batch_size=devices * mbs,
    )
    geneformer_config = GeneformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
    )
    model = GeneformerModel(geneformer_config, tokenizer=tokenizer)

    trainer.fit(model, data)
    checkpoint_path = Path(trainer.logger.log_dir) / "ckpt"
    trainer.save_checkpoint(checkpoint_path)
