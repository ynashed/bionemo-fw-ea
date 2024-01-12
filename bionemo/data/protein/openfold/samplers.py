# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import random
from collections import Counter, deque
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import Sampler

from bionemo.data.protein.openfold.datasets import (
    FinetuningDataset,
    InitialTrainingDataset,
    ValidationDataset,
)
from bionemo.data.protein.openfold.helpers import get_seed_from_string


class InitialTrainingSampler(Sampler[Tuple[int, int]]):
    """Sampler for initial training dataset."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        device_batch_size: int,
        global_batch_size: int,
        num_train_iters: int,
        seed: int,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
        num_prev_iters: int,
    ) -> None:
        assert num_prev_iters <= num_train_iters
        if is_distributed:
            assert rank is not None
            assert world_size is not None
            assert global_batch_size % world_size == 0
        weights = dataset.get_sampler_weights()
        num_samples_in_device_epoch = num_train_iters * device_batch_size
        num_samples_in_global_epoch = num_train_iters * global_batch_size
        # Sample indices:
        index_generator = torch.Generator()
        index_generator.manual_seed(seed)
        random_indices = torch.multinomial(
            input=weights,
            num_samples=num_samples_in_global_epoch,
            replacement=True,
            generator=index_generator,
        )
        # Sample seeds:
        seed_generator = torch.Generator()
        seed_generator.manual_seed(seed)
        random_seeds = torch.randint(
            low=0,
            high=2**63 - 1,
            size=[num_samples_in_global_epoch],
            generator=seed_generator,
        )
        # Create (index, seed) pairs:
        assert random_indices.size() == random_seeds.size()
        indices = random_indices.tolist()
        seeds = random_seeds.tolist()
        assert len(indices) == len(seeds)
        index_seed_pairs = list(zip(indices, seeds))
        if is_distributed:
            index_seed_pairs = index_seed_pairs[rank::world_size]
        assert len(index_seed_pairs) == num_samples_in_device_epoch
        # Move forward by skipping previous iterations:
        offset = num_prev_iters * device_batch_size
        assert offset <= len(index_seed_pairs)
        self.index_seed_pairs = index_seed_pairs[offset:]

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        assert hasattr(self, "index_seed_pairs")
        yield from self.index_seed_pairs
        # del self.index_seed_pairs # aCHCK

    def __len__(self) -> int:
        assert hasattr(self, "index_seed_pairs")
        return len(self.index_seed_pairs)


class ValidationSampler(Sampler[Tuple[int, int]]):
    """Sampler for validation dataset."""

    def __init__(
        self,
        dataset: ValidationDataset,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
    ) -> None:
        dataset_length = len(dataset)  # 180
        if is_distributed:
            assert rank is not None
            assert world_size is not None
            epoch_length = math.ceil(dataset_length / world_size)
        else:
            epoch_length = dataset_length
        seeds = [get_seed_from_string(pdb_chain_id) for pdb_chain_id in dataset.pdb_chain_ids]
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self._dataset_length = dataset_length
        self._epoch_length = epoch_length
        self._seeds = seeds
        indices = list(range(self._dataset_length))
        self.indices = indices[self.rank :: self.world_size]
        self.seeds = [self._seeds[index] for index in indices]

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        yield from zip(self.indices, self.seeds)

    def __len__(self) -> int:
        return len(self.indices)  # self._epoch_length


class FinetuningSampler(Sampler[Tuple[str, int, int]]):
    """Sampler for the fine-tuning dataset."""

    def __init__(
        self,
        dataset: FinetuningDataset,
        dataset_weights: Dict[str, float],
        device_batch_size: int,
        global_batch_size: int,
        num_train_iters: int,
        seed: int,
        is_distributed: bool,
        rank: Optional[int],
        world_size: Optional[int],
        num_prev_iters: int,
    ) -> None:
        assert num_prev_iters <= num_train_iters
        if is_distributed:
            assert rank is not None
            assert world_size is not None
            assert global_batch_size % world_size == 0
        weights = dataset.get_sampler_weights()
        num_samples_in_global_epoch = num_train_iters * global_batch_size
        num_samples_in_device_epoch = num_train_iters * device_batch_size
        # Sample datasets:
        dataset_rng = random.Random(seed)
        assert sum(dataset_weights.values()) > 0
        dataset_choices = dataset_rng.choices(
            population=list(dataset_weights.keys()),
            weights=list(dataset_weights.values()),
            k=num_samples_in_global_epoch,
        )
        # Sample indices:
        dataset_choices_cnt = Counter(dataset_choices)
        index_generators = {k: torch.Generator() for k in weights.keys()}
        for index_generator in index_generators.values():
            index_generator.manual_seed(seed)
        indices = {
            k: torch.multinomial(
                input=weights[k],
                num_samples=dataset_choices_cnt[k],
                replacement=True,
                generator=index_generators[k],
            )
            for k in index_generators.keys()
            if dataset_choices_cnt[k] > 0
        }
        indices = {k: deque(tensor.tolist()) for k, tensor in indices.items()}
        indices = [indices[dataset_choice].popleft() for dataset_choice in dataset_choices]
        # Sample seeds:
        seed_generator = torch.Generator()
        seed_generator.manual_seed(seed)
        seeds = torch.randint(
            low=0,
            high=2**63 - 1,
            size=[num_samples_in_global_epoch],
            generator=seed_generator,
        )
        seeds = seeds.tolist()
        # Create (dataset, index, seed) triples:
        assert len(dataset_choices) == len(indices) == len(seeds)
        dataset_index_seed_triples = list(zip(dataset_choices, indices, seeds))
        if is_distributed:
            dataset_index_seed_triples = dataset_index_seed_triples[rank::world_size]
        assert len(dataset_index_seed_triples) == num_samples_in_device_epoch
        # Move forward by skipping previous iterations:
        offset = num_prev_iters * device_batch_size
        assert offset <= len(dataset_index_seed_triples)
        self.dataset_index_seed_triples = dataset_index_seed_triples[offset:]

    def __iter__(self) -> Iterator[Tuple[str, int, int]]:
        assert hasattr(self, "dataset_index_seed_triples")
        yield from self.dataset_index_seed_triples
        del self.dataset_index_seed_triples

    def __len__(self) -> int:
        assert hasattr(self, "dataset_index_seed_triples")
        return len(self.dataset_index_seed_triples)
