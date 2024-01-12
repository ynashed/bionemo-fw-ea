# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Callable, Iterable, Iterator, List, Optional, Union

from nemo.utils import logging
from torch.utils.data.sampler import Sampler


class SizeAwareBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices. Keeps track of
    the some total size of the mini-batch (eg number of nodes in
    the resulting graph), rather than the number of elements in the mini-batch.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        max_total_size (int or float): Max total size of the resulting mini-batch
          calculated as a sum of per-element sizes, will be disobeyed if only one element in a batch
        sizes (Iterable or Callable): sizes[i] should be a size of i'th element in the dataset (eg number of nodes in the graph)
            if callable, will take a sample, and return the size
        batch_size (int, optional): average number of samples in one batch. Default to be None.
            batch_size and num_batches can't be set as None at the same time.
        num_batches (int, optional): number of batches to yield, set to be around len(sampler) // batch_size. Default to be None.
            batch_size and num_batches can't be set as None at the same time.
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        max_total_size: Union[int, float],
        sizes: Union[Iterable[Union[int, float]], Callable],
        batch_size: Optional[int] = None,
        num_batches: Optional[int] = None,
        **kwargs,
    ) -> None:
        if not isinstance(max_total_size, (int, float)) or max_total_size <= 0:
            raise ValueError("max_total_size should be a positive number, " f"but got max_total_size={max_total_size}")

        if batch_size is None and num_batches is None:
            raise ValueError(
                "One of batch_size and num_batches should be a positive integer value, "
                "but got None for both of them"
            )
        elif num_batches is None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size should be a positive integer value, " f"but got batch_size={batch_size}")
            num_batches = (len(sampler) // batch_size) + (len(sampler) % batch_size != 0)
        elif batch_size is None:
            if not isinstance(num_batches, int) or num_batches <= 0:
                raise ValueError(
                    "num_batches should be a positive integer value, " f"but got num_batches={num_batches}"
                )
            batch_size = len(sampler) // num_batches

        if callable(sizes):
            self.sizes_cache = {}
            self.callable_sizes = True
        else:
            self.callable_sizes = False
            if isinstance(sizes, dict):
                max_size = max(sizes.values())
                min_size = min(sizes.values())
            else:
                max_size = max(sizes)
                min_size = min(sizes)

            if max_size > max_total_size:
                logging.warning(
                    "Sizes of some elements in the dataset exceed max_total_size "
                    f"{max_total_size}. Such elements will be skipped. max(sizes) = {max_size}"
                )
            if min_size > max_total_size:
                raise RuntimeError(
                    f"Minimum element size in the dataset exceeds "
                    f"requested max_total_size ({min_size} > {max_total_size}). "
                    f"No samples can be generated."
                )
        if len(kwargs) > 0:
            logging.warning(
                'SizeAwareBatchSampler got the following unsupported arguments ' f'that will be ignored: {kwargs}'
            )

        self.sampler = sampler
        self.max_total_size = max_total_size
        self.sizes = sizes
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.dataset = sampler
        if hasattr(sampler, "dataset"):
            self.dataset = sampler.dataset
        elif hasattr(sampler, "data_source"):
            self.dataset = sampler.data_source

    def set_epoch(self, epoch: int) -> None:
        try:
            self.sampler.set_epoch(epoch)
        except AttributeError:
            logging.error(f"Trying to call set_epoch on {self.sampler.__class__} which does not have such method")

    def __len__(self):
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        batch_total_size = 0
        yield_num_batches = 0
        batch = []
        counter = 1
        num_samples = 0
        while yield_num_batches < self.num_batches:
            if counter > 1:
                logging.warning(
                    f"Sampler is being used {counter} times in one epoch, try to increase batch_size = {self.batch_size}, "
                    f"yield_num_batches = {yield_num_batches}, num_batches = {self.num_batches}"
                )
            for idx in self.sampler:
                if self.callable_sizes:
                    if idx in self.sizes_cache:
                        new_size = self.sizes_cache[idx]
                    else:
                        new_size = self.sizes(self.dataset[idx])
                        self.sizes_cache[idx] = new_size
                else:
                    new_size = self.sizes[idx]
                if new_size > self.max_total_size:
                    logging.debug(
                        f'Size of element {idx} exceeds max_total_size'
                        f' ({new_size} > {self.max_total_size}), skipping'
                    )
                    num_samples += 1
                    continue
                if new_size + batch_total_size > self.max_total_size:
                    num_samples += len(batch)
                    yield batch
                    yield_num_batches += 1
                    batch_total_size = 0
                    batch = []
                batch.append(idx)
                batch_total_size += new_size
                if yield_num_batches >= self.num_batches:
                    if num_samples < len(self.sampler):
                        logging.warning(
                            f"Only {num_samples} out of {len(self.sampler)} samples are used in one epoch, "
                            f"try to increase num_batches = {self.num_batches} or reduce batch_size = {self.batch_size}"
                        )
                    break
            counter += 1
            if self.callable_sizes and num_samples == 0:
                raise RuntimeError(
                    f"Minimum element size in the dataset exceeds "
                    f"requested max_total_size ({min(list(self.sizes_cache.values()))} > {self.max_total_size}). "
                    f"No samples is generated."
                )

        # return the last one
        if len(batch) > 0 and yield_num_batches < self.num_batches:
            num_samples += len(batch)
            yield batch
            yield_num_batches += 1

        if num_samples < len(self.sampler):
            logging.warning(
                f"Only {num_samples} out of {len(self.sampler)} samples are used "
                f"try to increase max_total_size = {self.max_total_size} or num_batches = {self.num_batches}"
            )

    @property
    def micro_batch_size(self):
        return self.batch_size

    @property
    def drop_last(self):
        return False
