# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
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


# TODO(@mgreaves, @jstjohn, @jomitchell) Consider different abstractions for pretraining, inference, and fine-tuning and see
#  how they would address code duplication in the case of ESM2+Geneformer as well as a third hypothetical model that does
#  not share the same types/loaders, such as OpenFold. The design should be flexible enough to allow for those differeht
#  use cases and not hide too much complexity that a user would want to customize, while reducing code duplication
#  between scripts.

import math
from pathlib import Path
from typing import Optional

from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.identity_op import IdentityOp
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import io
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.utils import logging
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from torch.nn import functional as F

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.lightning import LossLoggingCallback
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.loss import BERTMLMLossWithReduction
from bionemo.llm.utils.logger_utils import WandbLoggerOptions, setup_nemo_lightning_logger


def get_config(
    seq_length: int,
    precision: str,
):
    geneformer_config = GeneformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp32_residual_connection=False,  # TODO(@jstjohn) check this
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
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
        qk_layernorm=False,  # TODO(@jstjohn) check this
        apply_residual_connection_post_layernorm=False,  # False is new default, True was BERT pub.
        bias_activation_fusion=True,  # TODO(@jstjohn) check this
        bias_dropout_fusion=True,  # TODO(@jstjohn) check this
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,  # This has to be set to True if we use the mixed precision plugin
        biobert_spec_option="bert_layer_local_spec",  # biobert_spec_option,
        nemo1_ckpt_path=None,  # nemo1_init_path,
    )
    return geneformer_config


class PassthroughHead(IdentityOp):
    def forward(self, x, *args, **kwargs):
        x["token_logits"] = x["token_logits"] ** 2
        return x


class CustomLossFn(BERTMLMLossWithReduction):
    def forward(self, batch, forward_out):
        loss = super().forward(batch, forward_out)
        new_loss = (loss[0] / 2, loss[1])
        new_loss[1]["avg"] = new_loss[1]["avg"] / 2
        return new_loss


class FinetuneTransform:
    def __call__(self, model: BioBertLightningModule):
        # change the head on the MegatronBioBertModel
        model.module.module.module.module.finetuning_head = PassthroughHead()  # TODO needs unrolling
        # change the Loss on the BioBertLightningModule
        model.module.loss_reduction_class = CustomLossFn
        # TODO add peft?
        # TODO freeze/unfreeze any additional parameters?
        return model


def main(
    ckpt_path: str,
):
    """
    Main function for the script.

    This script takes a pre-traineed GPT model and finetunes it on an arbitrary dataset with a
    custom head. The custom head is a simple linear layer that takes the last hidden state of the
    BERT model and outputs a single scalar value. The script is meant to be a starting point for
    users who want to finetune a BERT model on their own dataset.

    The broad steps are:
    1. Create the dataset module
    2. Load the pre-trained BERT model (TODO: load via bare io call?)
        a. right now let's ignore the complexity of loading weights, and just assume we have a model
    3. Create a custom head
    4. Attach the custom head to the BERT model
    5. Add a PEFT method to the model
    6. Train the model

    A couple notes:
    * Since the head initialization is implemented as a callback, we must ensure that the head
        is only attached to the model for a new fine-tuning job. If resuming a fine-tuning job, the head
        should be loaded from the checkpoint instead.
    * We should be able to do progessive fine-tuning whereby we first train one head, then add a new head
        and train the model with both heads or just the second head, etc.

    """
    # Everything below this line can eventually be added as a command-line arg
    result_dir: Path = Path("results")
    data_dir: Path = Path("/workspaces/bionemo-github/test_data/cellxgene_2023-12-15_small/processed_data")
    # wandb_project: Optional[str] = None
    devices: int = 1
    num_steps: int = 100
    limit_val_batches: float | int = 2
    val_check_interval: int = 10
    num_nodes: int = 1
    precision: str = "bf16-mixed"
    seq_length: int = 128
    micro_batch_size: int = 32
    num_dataset_workers: int = 0
    lr: float = 1e-3
    cosine_rampup_frac: float = 0.01
    cosine_hold_frac: float = 0.05
    experiment_name: str = "geneformer-finetune"
    create_tensorboard_logger: bool = False

    # Everything above this line can be a command-line arg

    # Create the result directory if it does not exist.
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup train/test/val data paths
    train_data_path = data_dir / "train"
    val_data_path = data_dir / "val"
    test_data_path = data_dir / "test"

    # Setup the strategy and trainer
    pipeline_model_parallel_size = 1
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
    )

    wandb_options: Optional[WandbLoggerOptions] = (
        None
        # if wandb_project is None
        # else WandbLoggerOptions(
        #     offline=wandb_offline,
        #     project=wandb_project,
        #     entity=wandb_entity,
        #     log_model=False,
        # )
    )
    trainer = nl.Trainer(
        devices=devices,
        max_steps=num_steps,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=val_check_interval,  # TODO(@jstjohn) Checkpoint saving is currently broken, fix and change this.
        num_nodes=num_nodes,
        callbacks=[
            # TODO(@skothenhill-nv) these need to be cleaned up when we have the automatic addition of track_io
            io.track_io(LossLoggingCallback)(),
            io.track_io(RichModelSummary)(max_depth=4),
            io.track_io(LearningRateMonitor)(),
        ],
        plugins=nl.MegatronMixedPrecision(precision=precision, amp_O2=False),
    )

    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")

    # Configure the data module and model
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=train_data_path,
        val_dataset_path=val_data_path,
        test_dataset_path=test_data_path,
        random_token_prob=0.02,  # changed to represent the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=micro_batch_size,
        global_batch_size=micro_batch_size * int(num_nodes * devices / pipeline_model_parallel_size),
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )

    geneformer_config = get_config(seq_length=seq_length, precision=precision)

    model = BioBertLightningModule(
        geneformer_config,
        tokenizer=tokenizer,
        optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                # TODO(@jstjohn) try decoupled_lr
                optimizer="adam",
                use_distributed_optimizer=True,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                # minimum learning rate is 1/100th of the initial learning rate, so eg lr=1e-3 -> min_lr=1e-5
                min_lr=lr / 100,
                warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
                interval="step",
                monitor="val_loss",
                constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
            ),
        ),
    )

    # # Configure our custom Checkpointer
    # checkpoint_callback = nl_callbacks.ModelCheckpoint(
    #     save_best_model=save_best_checkpoint,
    #     save_last=save_last_checkpoint,
    #     monitor=metric_to_monitor_for_checkpoints,  # "val_loss",
    #     save_top_k=save_top_k,
    #     every_n_train_steps=save_every_n_steps,
    #     enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
    #     async_save=False,  # Tries to save asynchronously, previously led to race conditions.
    # )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=result_dir,
        name=experiment_name,
        initialize_tensorboard_logger=create_tensorboard_logger,
        wandb_kwargs=wandb_options,
        # ckpt_callback=checkpoint_callback,
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        model_transform=FinetuneTransform(),
        # resume=resume.AutoResume(
        #     path=restore_from_checkpoint_path,  # Overrides the path found by resume_if_exists when set.
        #     resume_if_exists=resume_if_exists,  # Looks for the -last checkpoint to continue training.
        #     resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        # ),
    )


if __name__ == "__main__":
    ckpt_path = "results/test_experiment/2024-08-08_16-52-30/checkpoints/test_experiment--val_loss=10.0553-epoch=0"
    main(ckpt_path)
