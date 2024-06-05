# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

from nemo import lightning as nl
from nemo.collections import llm


devices, seq_length = 1, 2048

strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
trainer = nl.Trainer(
    devices=devices,
    max_steps=5,
    accelerator="gpu",
    strategy=strategy,
)

data = llm.MockDataModule(seq_length=seq_length, global_batch_size=32)

gpt_config = llm.GPTConfig(
    num_layers=4,
    hidden_size=4096,
    ffn_hidden_size=4096,
    num_attention_heads=32,
    seq_length=seq_length,
)
model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)

trainer.fit(model, data)
checkpoint_path = Path(trainer.logger.log_dir) / "ckpt"
trainer.save_checkpoint(checkpoint_path)
