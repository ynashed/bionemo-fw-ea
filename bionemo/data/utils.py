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
from typing import List, Dict, Optional, Any
from enum import Enum
import re
import braceexpand
import os
from omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.utils import logging
from bionemo.data.molecule import MoleculeCsvDataset
from nemo.core import Dataset
from typing import Optional
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import (
    get_samples_mapping,
    # TODO remove when upgraded to NeMo 1.13
    make_text_memmap_bin_compatibility,
)

__all__ = [
    'DatasetTypes',
    'expand_dataset_paths',
    'create_dataset',
    'build_train_valid_test_datasets',
    'handle_index',
    'MappedDataset',
    'SliceDataset',
    'NeMoUpsampling',
]

class DatasetTypes(Enum):
    zinc_csv  = 0


def expand_dataset_paths(filepath: str, ext: str) -> List[str]:
    """Expand dataset paths from braces"""
    filepath = filepath + ext if ext else filepath
    # TODO this should eventually be moved to a Nemo fileutils module or similar
    filepath = re.sub(r"""\(|\[|\<|_OP_""", '{', filepath) # replaces '(', '[', '<' and '_OP_' with '{'
    filepath = re.sub(r"""\)|\]|\>|_CL_""", '}', filepath) # replaces ')', ']', '>' and '_CL_' with '}'
    dataset_paths = list(braceexpand.braceexpand(filepath))
    return dataset_paths

def check_paths_exist(dataset_paths):
    """Check that the expanded dataset paths are valid and they exist."""
    errors = []
    for filepath in dataset_paths:
        if not os.path.exists(filepath):
            errors.append(filepath)
    return errors

class DatasetBuilderSpec(object):
    """
    Abstract class for building datasets.

    Constructing a dataset from a list of files is a complex task.

    It involves:
    1. Parsing the string that delineates the files to be included, e.g.,
        "x001.csv,x002.csv" such that it corresponds to a list of files.
    2. Ensuring that the listed files actually exist on the filesystem.
    3. Instantiating a dataset from the given files.

    In each of these steps, there are ways that different types of files
    that could be used, different parsing schemes, and different PyTorch
    datasets that may be used.

    A DatasetBuilder encapsulates each of these steps such that the dataset
    instantiation can be performed by any class that implements the methods
    here.

    """
    def __init__(self, options: Dict[str, Any]):
        """
        Initializes a dataset builder

        Arguments:
            options (Dict[str, Any]): A dictionary that stores attributes that
                can arbitrarily be accessed by the builder.

        Attributes:
            options (Dict[str, Any]): Stores attributes that can be accessed
                by the builder
            dataset_paths (List[str]): Corresponds to the files to include
                in the dataset
            dataset (Optional[Dataset]): This will be the dataset that has been
                constructed by the builder, if it has been built yet.
        """
        self.options = options
        self.dataset_paths = []
        self.dataset = None

    def initialize_dataset(self):
        """
        Initializes the dataset

        Returns:
            Dataset: The initialized dataset
        """
        self.format_dataset_paths()
        self.check_dataset_paths_exist()
        return self.create_dataset()

    def check_dataset_paths_exist(self):
        """
        Checks whether the dataset's paths exist.

        Raises:
            AssertionError: If there are any files in the dataset's paths
                that do not exist.

        """
        errors = []
        for filepath in self.dataset_paths:
            file_does_not_exist = self.check_path(filepath)
            if file_does_not_exist:
                errors.append(file_does_not_exist)
        if len(errors) != 0:
            raise ValueError(
                f"Following files do not exist {' '.join(errors)}"
            )

    def check_path(self, filepath):
        """
        Checks whether an individual file exists. Because Dataset's may perform
        their own modifcations on filepath names to read/write data, the
        logic to check whether the file exists may differ between the concrete
        dataset classes.

        Arguments:
            filepath (str): a string that can be used to identify the filepath

        Returns:
            Optional[str]: If the file exists, this returns None, otherwise
                it returns the on the filepath.
        """
        if not os.path.exists(filepath):
            return filepath

    def format_dataset_paths(self):
        """
        Modifies the objects dataset_paths attribute so that it contains
        the paths of all of the files to include in the dataset.

        This function will essentially allow for custom parsing of arbitrary
        input text.

        Examples:
        * 'chr1,chr2' -> ['chr1.fa', 'chr2.fa']
        * 'x001..x003' -> ['x001.csv', 'x002'csv', 'x003.csv']

        """
        raise NotImplementedError()

    def create_dataset(self):
        """
        This instantiates a concrete dataset class and sets it to this object's
        `dataset` attribute.
        """
        raise NotImplementedError()

def get_filepath(options):
    filepath = cfg_get_key(options['cfg'], 'dataset_path', default='')
    filepath = os.path.join(
        filepath, options['name'], options['dataset'])

    return filepath


class CSVDatasetBuilder(DatasetBuilderSpec):
    """
    Builds instances MoleculeCsvDataset

    Examples:
    >>> dataset_builder = CSVDatasetBuilder(options=...)
    >>> csv_dataset_instance = dataset_builder.initialize_dataset()

    """
    def __init__(self, options: Dict[str, Any]):
        """
        Initializes a dataset builder

        Arguments:
            options (Dict[str, Any]): A dictionary that stores attributes that
                can arbitrarily be accessed by the builder. Requires keys:
                ['filepath', 'cfg'].

        Attributes:
            options (Dict[str, Any]): Stores attributes that can be accessed
                by the builder
            dataset_paths (List[str]): Corresponds to the files to include
                in the dataset
            dataset (Optional[Dataset]): This will be the dataset that has been
                constructed by the builder, if it has been built yet.
        """
        super().__init__(options)

    def format_dataset_paths(self):
        """
        Parses CSV paths.

        """
        filepath = get_filepath(self.options)
        # Get datasets and load data
        logging.info(f'Loading data from {filepath}')
        self.dataset_paths = expand_dataset_paths(
            filepath, ".csv")

    def check_path(self, filepath):
        """
        Checks whether a CSV exists.

        Arguments:
            filepath (str): a string that can be used to identify the filepath

        Returns:
            Optional[str]: If the file exists, this returns None, otherwise
                it returns the on the filepath.

        """
        if not os.path.exists(filepath):
            return filepath

    def create_dataset(self):
        """
        Instantiates a MoleculeCsvDataset.

        Returns:
            Dataset: Dataset instantiated from paths.
        """
        cfg = self.options['cfg']
        self.dataset = MoleculeCsvDataset(
            dataset_paths=self.dataset_paths, cfg=cfg)
        return self.dataset


class DatasetFactorySpec(object):

    def create_dataset(self, options) -> DatasetBuilderSpec:
        """
        Creates a dataset with the given options.
        """
        raise NotImplementedError()


class FormattedDatasetFactory(DatasetFactorySpec):
    """
    Abstract class for constructing datasets builders.

    Attributes:
        formats (Dict[str, DatasetBuilderSpec]): Keys are formats that this
            factory supports, values are the classes of the builders.

    """
    formats: Dict[str, DatasetBuilderSpec]

    def create_dataset(self, options) -> DatasetBuilderSpec:
        """
        Creates a dataset with the given options.

        Arguments:
            options (Dict[str, Any]): Requires key 'cfg' to have a key for
                'dataset_format', which refers to a format to use from
                this objects `formats` attribute. Additional key
                requirements will vary by the dataset being built.

        Returns:
            Dataset: the initialized dataset to use.

        Raises:
            ValueError: if 'dataset_format' from options['cfg'] is not
                recognized.

        """

        dataset_format = cfg_get_key(options['cfg'], 'dataset_format')

        if dataset_format in self.formats:
            builder_cls = self.formats[dataset_format]
            builder = builder_cls(options)
        else:
            raise ValueError(f"Unrecognized data format."
                             f" Expected one of: {self.keys()}")
        return builder.initialize_dataset()

    @classmethod
    def keys(cls):
        """Lists available format values.

        Returns:
            List[str]: Allowable format values.
        """
        return list(cls.formats.keys())


class DefaultDatasetFactory(FormattedDatasetFactory):
    """
    Examples:
    >>> dataset_factory = DefaultDatasetFactory()
    >>> dataset = dataset_factory.create_dataset(options={'datset_format': 'csv', ...})
    """

    def __init__(self):
        """
        Initializes a dataset factory for handling csv and bin formats.
        """
        self.formats = {
            'csv': CSVDatasetBuilder,
        }


def get_default_builder_factory():
    """
    Returns:
        DatasetFactory: A factory method for producing datasets.
    """
    return DefaultDatasetFactory()


def create_dataset(
    cfg: DictConfig,
    num_samples: int,
    name: str,
    dataset: str,
    dataset_factory: Optional[DatasetFactorySpec] = None,
):
    if dataset_factory is None:
        dataset_factory = get_default_builder_factory()

    options = {
        'cfg': cfg,
        'num_samples': num_samples,
        'name': name,
        'dataset': dataset,
    }

    dataset = dataset_factory.create_dataset(options)

    return dataset


def build_train_valid_test_datasets(
    cfg: DictConfig,
    trainer: Trainer,
    train_valid_test_num_samples: List[int],
    dataset_factory: Optional[FormattedDatasetFactory] = None,
):

    ds_train = cfg.dataset.train
    ds_val = cfg.dataset.val
    ds_test = cfg.dataset.test

    # Build individual datasets.
    train_dataset = create_dataset(cfg, train_valid_test_num_samples[0],
                                   'train', ds_train,
                                   dataset_factory,
                                   )

    validation_dataset = create_dataset(cfg, train_valid_test_num_samples[1],
                                        'val', ds_val,
                                        dataset_factory,
                                        )

    test_dataset = create_dataset(cfg, train_valid_test_num_samples[2],
                                  'test', ds_test,
                                  dataset_factory,
                                  )

    return (train_dataset, validation_dataset, test_dataset)

def cfg_get_key(cfg, key, default=None):
    with open_dict(cfg):
        return cfg.get(key, default)


def handle_index(dataset, idx):
    """
    Remaps negative indices and handles numpy int indices.

    Arguments:
        dataset (Dataset): dataset to index into
        idx (int): Index. Can include negative indices.
    Returns:
        int: Remapped and fully qualified index.

    Raises:
        IndexError: If a negative index is out of range.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from torch.utils.data import TensorDataset
        >>> from nemo_chem.data.fasta_dataset import handle_index
        >>> dataset = TensorDataset(torch.tensor(-np.arange(5)))
        >>> handle_index(dataset, 1)
        1
        >>> handle_index(dataset, -2)
        3

    """
    if idx < 0 and idx > -len(dataset) - 1:
        idx = len(dataset) + idx
    elif idx < 0:
        raise IndexError(f'Index out of range: {idx}')
    return idx


class SliceIndex:
    def __init__(self, dataset, start, end):
        if start < 0:
            raise ValueError(f'start must be > 0: {start}')
        if end < start:
            raise ValueError(f'end must be >= start: {end} not >= {start}')
        if end > len(dataset):
            raise ValueError(
                f'end must be <= dataset length: {end} not <= {len(dataset)}'
                )

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

        Arguments:
            dataset (Dataset): dataset to remap.
            num_samples (Optional[int]): Number of samples the dataset should
                contain. The sampling strategy is based on
                `create_sample_mapping`. `create_sample_mapping` must support
                `num_samples=None` in order for this `num_samples` to be None.
        """
        self._dataset = dataset
        self.sample_mapping = self.create_sample_mapping(dataset, num_samples)

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

    def __init__(self, dataset, num_samples=None, data_prefix=None,
                 max_seq_length=None, seed=None, cfg=None,
                 index_mapping_dir=None,
                 name=None,
                 ):
        self.data_prefix, self.max_seq_length, self.seed = _infer_kwarg_values(
            cfg, data_prefix, max_seq_length, seed
        )
        self.index_mapping_dir = index_mapping_dir
        self.name = name
        super().__init__(dataset, num_samples)

    def create_sample_mapping(self, dataset: Dataset, num_samples: int):
        if num_samples is None:
            num_samples = len(dataset)

        if num_samples == 0:
            raise ValueError('Number of samples is 0. Cannot be sampled.')

        # TODO remove when upgraded to NeMo 1.13
        make_text_memmap_bin_compatibility(dataset)
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
