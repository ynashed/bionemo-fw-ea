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


import hashlib
from typing import Any, Callable, Dict, List, Optional, Union

import torch


def float_or_int_or_none(value: Union[str, float, int, None]) -> Union[float, int, None]:
    """Converts a given value into a float, int, or None.

    Args:
        value (Union[str, float, int, None]): A value that can be either a string, float, int, or None.

    Returns:
        Union[float, int, None]: A float, int, or None based on the input value.

    If the input value is None or "None", it returns None.
    If the input value is an int or float, it returns the same value.
    If the input value is a string, it tries to convert it into an int if possible, otherwise into a float.
    """
    if value is None or value == "None":
        return
    if isinstance(value, (int, float)):
        return value
    if value.isdigit():
        return int(value)
    return float(value)


def parse_kwargs_to_arglist(kwargs: Dict[str, Any]) -> List[str]:
    """Converts a dictionary of keyword arguments into a list of command-line arguments.

    Args:
        kwargs (Dict[str, Any]): A dictionary where keys are argument names and values are argument values.

    Returns:
        A list of strings, where each string is a command-line argument in the format '--argument-name value'.
    """
    arglist = []
    for k, v in kwargs.items():
        arglist.extend([f"--{k.replace('_', '-')}", str(v)])
    return arglist


def infer_global_batch_size(
    micro_batch_size: int,
    num_nodes: int,
    devices: int,
    accumulate_grad_batches: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> int:
    """Infers the global batch size based on the micro batch size, number of nodes, devices, accumulation of gradient batches, and model parallel sizes.

    Args:
        micro_batch_size (int): The micro batch size.
        num_nodes (int): The number of nodes.
        devices (int): The number of devices.
        accumulate_grad_batches (int): The accumulation of gradient batches. Defaults to 1.
        tensor_model_parallel_size (int): The tensor model parallel size. Defaults to 1.
        pipeline_model_parallel_size (int): The pipeline model parallel size. Defaults to 1.

    Returns:
        int: The global batch size.
    """
    if not all(
        isinstance(arg, int)
        for arg in [
            micro_batch_size,
            num_nodes,
            devices,
            accumulate_grad_batches,
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
        ]
    ):
        raise ValueError(
            f"All arguments must be of type int, got {type(micro_batch_size)}, {type(num_nodes)}, {type(devices)}, "
            f"{type(accumulate_grad_batches)}, {type(tensor_model_parallel_size)}, and {type(pipeline_model_parallel_size)}"
        )
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be greater than 0, got {micro_batch_size}")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be greater than 0, got {num_nodes}")
    if devices <= 0:
        raise ValueError(f"devices must be greater than 0, got {devices}")
    if accumulate_grad_batches <= 0:
        raise ValueError(f"accumulate_grad_batches must be greater than 0, got {accumulate_grad_batches}")
    if tensor_model_parallel_size <= 0:
        raise ValueError(f"tensor_model_parallel_size must be greater than 0, got {tensor_model_parallel_size}")
    if pipeline_model_parallel_size <= 0:
        raise ValueError(f"pipeline_model_parallel_size must be greater than 0, got {pipeline_model_parallel_size}")
    if devices % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise ValueError(
            f"devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size, "
            f"got {devices} and {tensor_model_parallel_size} * {pipeline_model_parallel_size}"
        )

    world_size = num_nodes * devices
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size
    data_parallel_size = world_size // model_parallel_size
    global_batch_size = micro_batch_size * data_parallel_size * accumulate_grad_batches
    return global_batch_size


def tensor_hash(tensor: torch.Tensor, hash_func: Optional[Callable] = None) -> str:
    """Generates a hash for the given tensor using the specified hash function.

    Args:
        tensor (torch.Tensor): The input tensor to be hashed.
        hash_func (Optional[Callable]): An optional hash function to use. If None, defaults to SHA-256.

    Returns:
        str: The resulting hash string.

    If no hash function is provided, SHA-256 is used by default. The function first converts the tensor to
    a contiguous array on the CPU and then to bytes before hashing.
    """
    tensor_bytes = tensor.cpu().contiguous().numpy().tobytes()
    if hash_func is None:
        return hashlib.sha256(tensor_bytes).hexdigest()
    else:
        return hash_func(tensor_bytes)


def tensor_dict_hash(tensor_dict: Dict[str, torch.Tensor], hash_func: Optional[Callable] = None) -> str:
    """Generates a hash for the given tensor dictionary using the specified hash function.

    Args:
        tensor_dict (Dict[str, torch.Tensor]): The input tensor dictionary to be hashed.
        hash_func (Optional[Callable]): An optional hash function to use. If None, defaults to SHA-256.

    Returns:
        str: The resulting hash string.

    If no hash function is provided, SHA-256 is used by default. The function first converts the tensor to
    a contiguous array on the CPU and then to bytes before hashing.
    """
    hash_value = ""
    for k in sorted(tensor_dict):
        hash_value += tensor_hash(tensor_dict[k], hash_func)
    return hash_value


def infer_num_val_samples(limit_val_batches: Union[float, int, str, None], len_valid_ds: int, global_batch_size: int):
    """Infers the number of validation samples based on the limit_val_batches parameter, the length of the validation dataset,
    and the global batch size.

    Args:
        limit_val_batches (Union[float, int, str, None]): The limit on the number of validation batches. Can be a float
            between 0 and 1, an integer, a string, or None. If None, defaults to 1.0.
        len_valid_ds (int): The length of the validation dataset.
        global_batch_size (int): The global batch size.

    Returns:
        int: The number of validation samples.

    Raises:
        ValueError: If the limited number of validation samples is less than the global batch size, or if the
            limit_val_batches parameter is invalid.

    If limit_val_batches is a float between 0 and 1, the number of validation samples is inferred as a fraction of the
    validation dataset length. If limit_val_batches is an integer greater than or equal to 1, the number of validation
    samples is inferred as the product of limit_val_batches and global batch size. If limit_val_batches is None, it defaults
    to 1.0, indicating that all validation samples should be used.
    """
    limit_val_batches = (
        1.0 if limit_val_batches is None else limit_val_batches
    )  # validation data does not require upsampling
    if 0 < limit_val_batches <= 1.0 and isinstance(limit_val_batches, float):
        num_val_samples = int(len_valid_ds * limit_val_batches)
        if num_val_samples < global_batch_size:
            raise ValueError(
                "The limited number of val samples %s is less than the global batch size %s"
                % (num_val_samples, global_batch_size)
            )
    elif limit_val_batches >= 1 and isinstance(limit_val_batches, int):
        num_val_samples = int(limit_val_batches * global_batch_size)
    else:
        raise ValueError("Invalid choice of limit_val_batches size: %s" % limit_val_batches)

    return num_val_samples
