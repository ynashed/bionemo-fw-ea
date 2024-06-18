# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from contextlib import contextmanager

import apex
import pytorch_lightning as pl
import torch
from megatron.core import parallel_state
from nemo.utils import logging


"""
This package contains utilities for managing the state of distributed model parallelism in Megatron and Apex.

In general you should just use the context manager `distributed_model_parallel_state` to manage the state of
your test. This context manager will handle the setup and teardown of the distributed model parallel state for you.

Example usage:
```python

from bionemo.contrib.testing import megatron_parallel_state_utils

def my_test():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        # your test code that requires megatron/apex parallel state to be set up here
```

"""


def _reset_microbatch_calculator():
    """
    Resets _GLOBAL_NUM_MICROBATCHES_CALCULATOR in apex which is used in NeMo to initilised model parallel in
    nemo.collections.nlp.modules.common.megatron.megatron_init.initialize_model_parallel_for_nemo
    """
    apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def _dummy() -> None:
    return


def _teardown_apex_megatron_cuda():
    """
    Cleans GPU allocation and model and data parallel settings after usage of a model:
    - sets the global variables related to model and data parallelism to None in Apex and Megatron:.
    - releases all unoccupied cached GPU memory currently held by the caching CUDA allocator, see torch.cuda.empty_cache
    """
    torch.cuda.empty_cache()
    _reset_microbatch_calculator()
    parallel_state.destroy_model_parallel()


def _initialize_distributed_parallel_state(
    local_rank: int = 0,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_model_parallel_split_rank: int = 0,
    interactive: bool = False,
) -> None:
    # initialize pytorch DDP
    # if not interactive and not torch.distributed.is_initialized():
    if not torch.distributed.is_initialized():
        logging.info("pytorch DDP is not initialized. Initializing with pytorch-lightening...")
        trainer = pl.Trainer(devices=1, strategy='ddp' if not interactive else "auto", num_nodes=1)

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(_dummy, trainer=trainer)
        trainer.strategy.setup_environment()

    if not interactive and parallel_state.is_unitialized():
        logging.info("Megatron DDP is not initialized. Initializing...")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        )


@contextmanager
def distributed_model_parallel_state():
    """Context manager for handling creating and cleaning up distributed model parallel state for tests.
    Use like:
    with distributed_model_parallel_state():
        # your test code here
    # After the block your state is cleaned up.
    """
    try:
        _teardown_apex_megatron_cuda()
        _initialize_distributed_parallel_state()
        yield
    finally:
        _teardown_apex_megatron_cuda()
