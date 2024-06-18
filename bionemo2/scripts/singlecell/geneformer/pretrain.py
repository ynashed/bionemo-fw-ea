# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# TODO(@mgreaves, @jstjohn, @jomitchell) Consider different abstractions for pretraining, inference, and fine-tuning and see
#  how they would address code duplication in the case of ESM2+Geneformer as well as a third hypothetical model that does
#  not share the same types/loaders, such as OpenFold. The design should be flexible enough to allow for those differeht
#  use cases and not hide too much complexity that a user would want to customize, while reducing code duplication
#  between scripts.

import argparse
from pathlib import Path

from nemo import lightning as nl
from nemo.utils import logging
from torch.nn import functional as F

from bionemo.contrib.data.singlecell.datamodule import SingleCellDataModule
from bionemo.contrib.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.contrib.lightning import LossLoggingCallback
from bionemo.contrib.model.biobert.lightning import BioBertLightningModule
from bionemo.contrib.model.biobert.model import BioBertConfig
from bionemo.contrib.utils.dtypes import get_autocast_dtype


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain Geneformer with single cell data.')
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Path to the data base directory, for example this might be '
        '/workspace/bionemo2/data/cellxgene_2023-12-15_small',
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        required=False,
        default=1,
        help='Number of GPUs to use for training. Default is 1.',
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=10000,
        help="Number of steps to use for training. Default is 10000.",
    )
    parser.add_argument(
        "--num-dataset-workers",
        type=int,
        required=False,
        default=0,
        help="Number of steps to use for training. Default is 0.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        required=False,
        default=10000,
        help="Number of steps to use for training. Default is 10000.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        required=False,
        default=2,
        help="Number of steps to use for training. Default is 2.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=64,
        help="Micro-batch size. Global batch size is inferred from this.",
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    train_data_path = data_dir / "train"
    val_data_path = data_dir / "val"
    test_data_path = data_dir / "test"

    num_nodes, devices, seq_length = args.num_nodes, args.num_gpus, 2048

    pipeline_model_parallel_size = 1
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=False,
    )

    precision = "bf16-mixed"

    trainer = nl.Trainer(
        devices=devices,
        max_steps=args.num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=args.limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=args.val_check_interval,  # TODO(@jstjohn) Checkpoint saving is currently broken, fix and change this.
        num_nodes=num_nodes,
        callbacks=[LossLoggingCallback()],
        plugins=nl.MegatronMixedPrecision(precision=precision, amp_O2=False),
    )

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
    micro_batch_size = args.micro_batch_size
    # gpus = 1
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=train_data_path,
        val_dataset_path=val_data_path,
        test_dataset_path=test_data_path,
        random_token_prob=0.1,  # this is the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=micro_batch_size,
        global_batch_size=micro_batch_size * int(devices / pipeline_model_parallel_size),
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=args.num_dataset_workers > 0,
        pin_memory=False,
        num_workers=args.num_dataset_workers,
    )
    geneformer_config = BioBertConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=True,  # TODO(@jstjohn) check this
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=True,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,  # TODO(@jstjohn) check this
        fp16_lm_cross_entropy=False,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        gradient_accumulation_fusion=False,  # THIS BREAKS STUFF, leave False
        layernorm_zero_centered_gamma=False,  # TODO(@jstjohn) check this
        layernorm_epsilon=1.0e-12,
        activation_func=F.gelu,  # TODO(@jstjohn) check this
        qk_layernorm=True,  # TODO(@jstjohn) check this
        apply_residual_connection_post_layernorm=False,  # False is new default, True was BERT pub.
        bias_activation_fusion=True,  # TODO(@jstjohn) check this
        bias_dropout_fusion=True,  # TODO(@jstjohn) check this
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=True,  # This has to be set to True if we use the mixed precision plugin
    )

    # The lightning class owns a copy of the actual model, and a loss function, both of which are configured
    #  and lazily returned by the `geneformer_config` object defined above.
    model = BioBertLightningModule(geneformer_config, tokenizer=tokenizer)
    trainer.fit(model, data)
    checkpoint_path = Path(trainer.logger.log_dir) / "ckpt"
    trainer.save_checkpoint(checkpoint_path)

    # TODO(@jstjohn) Doesnt work right now. Get this working.
    # TypeError: `model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `MegatronParallel`
    results = trainer.predict(dataloaders=data.test_dataloader())
    print(results[0])
