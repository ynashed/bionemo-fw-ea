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

from typing import List, Dict, Optional, Any
from enum import Enum
import re
import braceexpand
import os
from copy import deepcopy
from omegaconf import DictConfig, open_dict
import torch.utils.data as pt_data
from pytorch_lightning.trainer.trainer import Trainer

from nemo.utils import logging
from bionemo.data.molecule import MoleculeCsvDataset

__all__ = ['DatasetTypes', 'expand_dataset_paths', 'check_paths_exist', 'create_dataset',
    'build_train_valid_test_datasets']

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
            'bin': BinDatasetBuilder,
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
