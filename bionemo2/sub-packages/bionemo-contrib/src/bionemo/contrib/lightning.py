# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.distributed
from megatron.core import parallel_state
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction, ReductionT


def get_dtype_device(torch_object) -> Tuple[torch.dtype, torch.device]:
    # TODO(@skothenhill): refafactor to use match statement
    if isinstance(torch_object, torch.Tensor):
        return torch_object.dtype, torch_object.device
    elif isinstance(torch_object, torch.nn.Module):
        # TODO(@skothenhill): handle, or more gracefully error, if parameters() is empty.
        return next(torch_object.parameters()).dtype, next(torch_object.parameters()).device
    elif isinstance(torch_object, dict):
        for torch_object in torch_object.values():
            if torch_object is not None:
                return get_dtype_device(torch_object)
    elif isinstance(torch_object, (list, tuple)):
        for i in range(len(torch_object)):
            if torch_object[i] is not None:
                return get_dtype_device(torch_object[i])
    else:
        raise ValueError(f"Unsupported type {type(torch_object)} in get_dtype_device")


# TODO(@jstjohn): Properly use the Generic for DataT and ReductionT usage. Define our own batch/output types.
def batch_collator(batches: Sequence[ReductionT]) -> ReductionT:
    """Takes a sequence of batches and collates them into a single batch.
        This is distinct from the standard pytorch default_collator since it does
        not add the batch dimension, it's assumed the batch
        dimension is already present in the input, as would be the case when
        parallelizing across minibatches.

    Args:
        batches (Sequence[ReductionT]): sequence of batches to collate into a single batch.

    Returns:
        A single batch of the same type as one of the elements of your input sequence.
    """
    # Class pattern matching (https://docs.python.org/3/reference/compound_stmts.html#class-patterns)
    # TODO (@skothenhill): Refactor to use the full pattern matching syntax rater than just on element 0.
    match batches[0]:
        case torch.Tensor():
            return torch.cat(batches, dim=0)
        case dict():
            return {key: batch_collator([batch[key] for batch in batches]) for key in batches[0]}
        case tuple():
            return tuple(batch_collator([batch[i] for batch in batches]) for i in range(len(batches[0])))
        case list():
            return [batch_collator([batch[i] for batch in batches]) for i in range(len(batches[0]))]
        case None:  # TODO(@jomitchell): Update the NeMo ReductionT to highlight that some keys can point to None
            # [None, None, None] -> None
            return None
        case _:
            raise ValueError(f"Unsupported type {type(batches[0])} in batch_collator")


# TODO(@jstjohn): Properly use the Generic for DataT and ReductionT usage. Define our own batch/output types.
class PassthroughLossReduction(MegatronLossReduction):
    """Internally in NeMo2.0 the forward step is always expected to return a loss reduction class, and forward is expected to return a loss.
    This class hijacks that mechanism to instead pass through the forward output unperturbed as the loss (to enable inference in the predict step), and then the
    reduce method is used to collate the batch of forward outputs into a single batch. This supports the model forward output being a tensor, dict, tuple,
    or list of tensors.
    """

    def forward(self, batch: DataT, forward_out: DataT) -> Tuple[torch.Tensor, DataT]:
        """_summary_

        Args:
            batch (DataT): The batch of data that was passed through the model to generate output.
            forward_out (torch.Tensor): The output from your model's forward pass.

        Returns:
            Tuple[torch.Tensor, ReductionT]: A tuple containing the loss tensor (dummy in this case) and the forward output (unmodified).
        """
        dtype, device = get_dtype_device(forward_out)
        return torch.zeros(1, device=device, dtype=dtype), forward_out

    def reduce(self, forward_out: Sequence[DataT]) -> DataT:
        """This overrides the standard reduce with a simplified version that just takes a list of your model's forward outputs
            and collates them togehter into a single output.

        Args:
            forward_out (Sequence[ReductionT]): _description_

        Returns:
            ReductionT: _description_
        """
        return batch_collator(forward_out)


class LightningPassthroughPredictionMixin:
    """A mixin that allows your model to do inference on the predict step by hijacking the nemo loss
    reduction mechanism and passing the model output through.
    """

    def predict_loss_reduction(self) -> PassthroughLossReduction:
        """For the predict step, pass through the forward pass output."""
        return PassthroughLossReduction()


class LossLoggingCallback(pl.Callback):
    def __init__(self):
        """Log the loss at the end of each batch. For training do not reduce across the epoch but do so for validation/test."""
        self.val_losses = []
        self.test_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Assuming the loss is computed internally and stored in pl_module
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            # TODO(@jstjohn): verify when the outputs are a dictionary of "loss" and when they are just one tensor value.
            if isinstance(outputs, dict):
                outputs = outputs['loss']
            # torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.AVG)
            loss = outputs
            pl_module.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # TODO(@jstjohn): Add a docstring with type hints for this lightning hook
        # Assuming the loss is computed internally and stored in pl_module
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            # TODO(@jstjohn): verify when the outputs are a dictionary of "loss" and when they are just one tensor value.
            if isinstance(outputs, dict):
                outputs = outputs['loss']
            # TODO verify that losses are already reduced across ranks
            # torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.AVG)
            loss = outputs
            self.test_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # TODO(@jstjohn): Add a docstring with type hints for this lightning hook
        # Assuming the loss is computed internally and stored in pl_module
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            # TODO(@jstjohn): verify when the outputs are a dictionary of "loss" and when they are just one tensor value.
            if isinstance(outputs, dict):
                outputs = outputs['loss']
            # TODO verify that losses are already reduced across ranks
            # torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.AVG)
            # TODO verify that losses are already reduced across ranks
            # torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.AVG)
            loss = outputs
            self.val_losses.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        # TODO(@jstjohn): Add a docstring with type hints for this lightning hook
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if len(self.val_losses) > 0:
                avg_val_loss = torch.stack(self.val_losses).mean()
                pl_module.log('val_loss', avg_val_loss, prog_bar=True, logger=True, rank_zero_only=True)
                self.val_losses.clear()

    def on_test_epoch_end(self, trainer, pl_module):
        # TODO(@jstjohn): Add a docstring with type hints for this lightning hook
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if len(self.test_losses) > 0:
                avg_test_loss = torch.stack(self.test_losses).mean()
                pl_module.log('test_loss', avg_test_loss, prog_bar=True, logger=True, rank_zero_only=True)
                self.test_losses.clear()
