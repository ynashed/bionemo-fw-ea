# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Sequence

import numpy as np
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.core import Dataset
from nemo.utils import logging

from bionemo.contrib.data.utils import handle_index


__all__ = ['MappedDataset', 'SliceDataset', 'ResamplingMappedDataset', 'FilteredMappedDataset']


class SliceIndex:
    def __init__(self, dataset, start, end):
        if start < 0:
            raise ValueError(f'start must be > 0: {start}')
        if end < start:
            raise ValueError(f'end must be >= start: {end} not >= {start}')
        if end > len(dataset):
            raise ValueError(f'end must be <= dataset length: {end} not <= {len(dataset)}')

        self.start = start
        self.end = end
        self.length = int(self.end - self.start)

    def __getitem__(self, idx):
        idx = handle_index(self, idx)
        return idx + self.start

    def __len__(self):
        return self.length


class MappedDataset(Dataset, ABC):
    def __init__(self, dataset: Dataset, num_samples: Optional[int] = None, consolidate_sample_mapping: bool = False):
        """
        Produces a remapped version of a `Dataset`.
        Can be used to create a subset of a dataset, or to shuffle it.
        Chainings of `MappedDataset` are supported and are collpased for efficiency.

        Arguments:
            dataset (Dataset): dataset to remap.
            num_samples (Optional[int]): Number of samples the dataset should
                contain. The sampling strategy is based on
                `create_sample_mapping`. `create_sample_mapping` must support
                `num_samples=None` in order for this `num_samples` to be None.
            consolidate_sample_mapping (bool): If True, the sample mapping will flatten any chained MappedDatasets
                                               (default: False since memory consumption will be high for large num_samples)
        """
        self._dataset = dataset

        self.sample_mapping = self.create_sample_mapping(dataset, num_samples)

        # consolidate sample mapping if dataset is MappedDataset
        if consolidate_sample_mapping and isinstance(dataset, MappedDataset):
            self.sample_mapping = [dataset.sample_mapping[i] for i in self.sample_mapping]
            self._dataset = dataset._dataset

    def __len__(self):
        return len(self.sample_mapping)

    def get_idx(self, idx):
        idx = self.sample_mapping[handle_index(self, idx)]
        idx = handle_index(self, idx)
        return idx

    def __getitem__(self, idx):
        idx = self.get_idx(idx)
        return self._dataset[idx]

    @abstractmethod
    def create_sample_mapping(self, dataset: Dataset, num_samples: int) -> Sequence[int]:
        """Sample mapping used to remap a dataset. Implemented by child class.

        Arguments:
            dataset (Dataset): dataset to discretize
            num_samples (int): Number of samples to include in the mapped
                dataset. Child classes may ignore if sampling is not enabled.

        Returns:
            sample_mapping (ArrayLike[int]): If `sample_mapping[i] == j`,
            the `i`th entry in this dataset will be `j`th entry of the original
            dataset.

        """
        raise NotImplementedError()


class SliceDataset(MappedDataset):
    def __init__(self, dataset: Dataset, start: int = 0, end: int = -1):
        """Slices a dataset on the fly.

        Args:
            dataset (Dataset): Dataset to slice
            start (int): First index of slice
            end (int): Last index of slice (exclusive)

        """
        self.start = handle_index(dataset, start)
        self.end = handle_index(dataset, end)
        super().__init__(dataset, None)

    def create_sample_mapping(self, dataset: Dataset, num_samples: Optional[int]) -> Sequence[int]:
        """Creates a sample mapping for trimming the `dataset` to length
        based on the slice arguments.

        Arguments:
            dataset (Dataset): Dataset to slice
            num_samples (Optional[int]): Ignored

        """
        return SliceIndex(dataset, self.start, self.end)


def _infer_kwarg_values(cfg, data_prefix, max_seq_length, seed):
    if data_prefix is None:
        data_prefix = cfg.get('data_prefix')

    if max_seq_length is None:
        max_seq_length = cfg.get('seq_length', None)
        if max_seq_length is None:
            raise ValueError('max_seq_length must be provided or in the config as data.seq_length')
        max_seq_length -= 2  # account for <CLS> / <EOS>

    if seed is None:
        seed = cfg.get('seed')

    if seed is None or max_seq_length is None or data_prefix is None:
        raise ValueError('seed, max_seq_length, and data_prefix must be provided or in the config')

    return data_prefix, max_seq_length, seed

    def create_sample_mapping(self, dataset, num_samples=None) -> Sequence[int]:
        return self.sample_map


class ResamplingMappedDataset(MappedDataset):
    """Upsamples / downsamples a dataset to a target length by repeating samples."""

    def __init__(
        self,
        dataset,
        num_samples=None,
        data_prefix=None,
        max_seq_length=None,
        seed=None,
        cfg=None,  # This is usually in model.data subfield
        index_mapping_dir=None,
        name=None,
    ):
        """
        Resamples a dataset to a specified target length by adjusting the sample distribution.
        This can involve either upsampling (repeating samples) or downsampling (reducing samples)
        to match the target sample size. It supports different strategies for resampling, such as
        in-memory mapping or online resampling, based on the provided configuration.

        Parameters:
            dataset (Dataset): The dataset to be resampled.
            num_samples (int, optional): The target number of samples after resampling. If not provided,
                the length of the dataset is used.
            data_prefix (str, optional): A prefix used for data processing or file naming. This can be
                inferred from the configuration if not provided directly. Used in memmap index mapping
                strategy.
            max_seq_length (int, optional): The maximum sequence length for samples in the dataset. This
                can also be inferred from the configuration.
            seed (int, optional): Seed for random number generation, used in shuffling the dataset. This
                can be provided directly or inferred from the configuration.
            index_mapping_dir (str, optional): Directory path where index mapping files are stored or will
                be generated. Used in memmap index mapping strategy.
            name (str, optional): An optional name for the dataset. If not provided, it is inferred from
                the data_prefix or set to None.
            cfg (dict, optional): A configuration dictionary, usually found in model.data, which can
                contain default values for other parameters if they are not explicitly provided. The following keys
                are used by this class:
                - `index_mapping_type`: Strategy for index mapping ("online" or "memmap").
                - `block_size`: Size of each block of samples in online sampling.
                - `cache_maxsize`: Maximum cache size for online sampling.
                - `shuffle`: Whether to shuffle the dataset during resampling.
                - `truncate_to_block_boundary`: Whether to truncate the sample mapping to block boundaries in online sampling.

        Raises:
            ValueError: If num_samples is set to 0, as it's not possible to resample to no samples.
            ValueError: If an unknown index_mapping_type is specified in the configuration.

        This class supports dynamic adjustment of the sample size of a dataset through different
        resampling strategies, enabling flexible dataset manipulation for training or evaluation purposes.
        """
        cfg = cfg or {}  # Convert a None cfg into a new empty dict
        self.data_prefix, self.max_seq_length, self.seed = _infer_kwarg_values(cfg, data_prefix, max_seq_length, seed)
        self.index_mapping_dir = index_mapping_dir
        self.name = name
        self.cfg = cfg

        super().__init__(dataset, num_samples)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int) -> Sequence[int]:
        if num_samples is None:
            num_samples = len(dataset)

        if num_samples == 0:
            raise ValueError('Number of samples is 0. Cannot be sampled.')

        # TODO: change default to 'online' once validated (and remove support in memmap?)
        self.cfg.get('index_mapping_type', 'memmap')

        # elif index_mapping_type == 'memmap':
        # TODO: remove support for memmap and use OnlineSampleMapping instead (more efficient)
        samples_mapping = get_samples_mapping(
            indexed_dataset=dataset,
            data_prefix=self.data_prefix,
            num_epochs=None,
            max_num_samples=num_samples,
            # account for <BOS> / <EOS>
            max_seq_length=self.max_seq_length,
            short_seq_prob=0.0,
            seed=self.seed,
            name=self.data_prefix.split('/')[-1] if self.name is None else self.name,
            binary_head=False,
            index_mapping_dir=self.index_mapping_dir,
            samples_mapping=None,
        )
        # truncate to max number of num_samples (None is all samples)
        samples_mapping = samples_mapping[:num_samples, 0]
        # else:
        #     raise ValueError(f'Unknown index_mapping_type: {index_mapping_type}, expected "online" or "memmap"')

        return samples_mapping


class FilteredMappedDataset(MappedDataset):
    """Filters samples from a dataset based on a criterion function by mapping the dataset samples."""

    def __init__(self, dataset, criterion_fn, num_samples=None):
        """
        Args:
            dataset (Dataset): Dataset to filter
            critetion_fn (Callable): Function that takes in a sample and returns True if the sample should be kept
        """
        self.criterion_fn = criterion_fn
        super().__init__(dataset=dataset, num_samples=num_samples)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        """Creates a sample mapping for filtering the `dataset` based on the criterion function."""
        samples_mapping = np.where(list(map(self.criterion_fn, dataset)))[0]
        ds_size = len(dataset)
        filtered_ds_size = len(samples_mapping)
        logging.debug(
            f"Filtered out (ignored) {ds_size - filtered_ds_size} samples ( {filtered_ds_size} / {ds_size} )"
        )

        # truncate to max number of num_samples (None is all samples)
        samples_mapping = samples_mapping[:num_samples]
        return samples_mapping


class IndexMappedDataset(MappedDataset):
    """Maps a dataset to a new dataset based on provides indices."""

    def __init__(self, dataset, indices, copy: bool = False):
        """
        Args:
            dataset (Dataset): Dataset to filter
            indices: indices to keep
            copy: Create a deep copy of the underlying dataset (can prevent UAF mistakes in DDP)
        """
        self.sample_mapping = indices
        super().__init__(dataset=deepcopy(dataset), num_samples=None)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        """Creates a sample mapping for filtering the `dataset` based on the criterion function."""
        return self.sample_mapping
