# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nemo import lightning as nl


def test_load_megatron_strategy():
    strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
    assert strategy.tensor_model_parallel_size == 1


def test_construct_nemo_lightning_trainer():
    trainer = nl.Trainer(
        devices=1,
        max_steps=5,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(tensor_model_parallel_size=1),
    )
    assert trainer.max_steps == 5
