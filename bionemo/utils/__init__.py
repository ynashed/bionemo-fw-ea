# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Sequence, Union


__all__: Sequence[str] = (
    "update_dataclass_config",
    "flatten_dict",
    "lookup_or_use",
)


def update_dataclass_config(cfg, dataset_config_class):
    """Update a dataset configuration with existing defaults"""
    default_cfg = asdict(dataset_config_class())
    default_cfg.update(cfg)
    return default_cfg


def flatten_dict(list_of_dicts):
    """Flatten a list of dictionaries to list without assuming all keys are identical"""
    flattened_dict = defaultdict(list)
    for metric_dict in list_of_dicts:
        for metric_name, metric_value in metric_dict.items():
            flattened_dict[metric_name].append(metric_value)

    return flattened_dict


def lookup_or_use(obj: Any, attr: Union[str, Any], *args, **kwargs):
    """Looks up an object from a module or returns the attribute if it is not a string"""
    # TODO [MG]: Why is this runtime reflection necessary? Can we avoid this indirection? Or is it absolutely vital?
    return getattr(obj, attr)(*args, **kwargs) if isinstance(attr, str) else attr
