#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from bionemo.data.equidock.db5_data import nemo_get_dataset


class Singleton(type):
    """
    Python metaclass to insures single instance of a class.

    Example Usage:
        class MySingleton(metaclass=Singleton)
            pass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DataManager(metaclass=Singleton):
    def __init__(self, cfg, **kwargs) -> None:
        self.train_ds = nemo_get_dataset(cfg.model.train_ds, 'train')
        self.validation_ds = nemo_get_dataset(cfg.model.validation_ds, 'val')
        self.test_ds = nemo_get_dataset(cfg.model.test_ds, 'test')

    def swap(self, cfg):
        """
        Override DataManager using a new cfg file
        """
        self.train_ds = nemo_get_dataset(cfg.model.train_ds, 'train')
        self.validation_ds = nemo_get_dataset(cfg.model.validation_ds, 'val')
        self.test_ds = nemo_get_dataset(cfg.model.test_ds, 'test')
