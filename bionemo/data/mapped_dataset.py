# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from abc import abstractmethod
from typing import Optional

import numpy as np
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.core import Dataset
from nemo.utils import logging

from bionemo.data.utils import handle_index


__all__ = ['MappedDataset', 'SliceDataset', 'NeMoUpsampling', 'FilteredMappedDataset']


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


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, num_samples: Optional[int] = None):
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
        """
        self._dataset = dataset
        self.sample_mapping = self.create_sample_mapping(dataset, num_samples)

        # consolidate sample mapping if dataset is MappedDataset
        if isinstance(dataset, MappedDataset):
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
    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
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

    def create_sample_mapping(self, dataset: Dataset, num_samples: Optional[int]):
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
        max_seq_length = cfg['seq_length'] - 2

    if seed is None:
        seed = cfg['seed']

    return data_prefix, max_seq_length, seed


class NeMoUpsampling(MappedDataset):
    """Upsamples a dataset to a target length by repeating samples."""

    def __init__(
        self,
        dataset,
        num_samples=None,
        data_prefix=None,
        max_seq_length=None,
        seed=None,
        cfg=None,
        index_mapping_dir=None,
        name=None,
    ):
        self.data_prefix, self.max_seq_length, self.seed = _infer_kwarg_values(cfg, data_prefix, max_seq_length, seed)
        self.index_mapping_dir = index_mapping_dir
        self.name = name
        super().__init__(dataset, num_samples)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        if num_samples is None:
            num_samples = len(dataset)

        if num_samples == 0:
            raise ValueError('Number of samples is 0. Cannot be sampled.')

        # map dataset samples to the desired num_samples
        samples_mapping = get_samples_mapping(
            indexed_dataset=dataset,
            data_prefix=self.data_prefix,
            num_epochs=None,
            max_num_samples=num_samples,
            # account for <BOS> / <EOS>
            max_seq_length=self.max_seq_length,
            short_seq_prob=0,
            seed=self.seed,
            name=self.data_prefix.split('/')[-1] if self.name is None else self.name,
            binary_head=False,
            index_mapping_dir=self.index_mapping_dir,
        )

        samples_mapping = samples_mapping[:num_samples, 0]
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
        logging.info(f"Filtered out (ignored) {ds_size - filtered_ds_size} samples ( {filtered_ds_size} / {ds_size} )")

        # truncate to max number of num_samples (None is all samples)
        samples_mapping = samples_mapping[:num_samples]
        return samples_mapping


class IndexMappedDataset(MappedDataset):
    """Maps a dataset to a new dataset based on provides indices."""

    def __init__(self, dataset, indices):
        """
        Args:
            dataset (Dataset): Dataset to filter
            indices: indices to keep
        """
        self.sample_mapping = indices
        super().__init__(dataset=dataset, num_samples=None)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        """Creates a sample mapping for filtering the `dataset` based on the criterion function."""
        return self.sample_mapping
