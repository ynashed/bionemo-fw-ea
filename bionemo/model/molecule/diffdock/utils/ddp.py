# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os

import torch
import torch.distributed as dist


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("not using a(ny) GPU(s)!")
    return device


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def reduce_tensor(tensor: torch.Tensor, num_gpus: int) -> torch.Tensor:
    if num_gpus > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        if rt.is_floating_point():
            rt = rt / num_gpus
        else:
            rt = rt // num_gpus
        return rt
    return tensor


def init_distributed(world_size) -> bool:
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"  # Needed for CUDA graphs
        dist.init_process_group(backend=backend)
        assert dist.is_initialized()

    if get_rank() == 0:
        logging.info(f"Distributed initialized. World size: {world_size}")
    return distributed


def get_rank() -> int:
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank
