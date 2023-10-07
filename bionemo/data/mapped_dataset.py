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
from functools import cached_property
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np
from typing import Optional
import random
import json
import os
from tqdm import tqdm

from bionemo.data.utils import handle_index

from nemo.utils import logging
from nemo.core import Dataset
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
import time

__all__ = [
    'MappedDataset',
    'SliceDataset',
    'NeMoUpsampling',
    'FilteredMappedDataset'
]

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
    def __init__(self, dataset: Dataset, num_samples: Optional[int] = None, consolidate_sample_mapping=True):
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
        if consolidate_sample_mapping and isinstance(dataset, MappedDataset):
            self.sample_mapping = list(map(lambda i: dataset.sample_mapping[i], self.sample_mapping))
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


def name_index_file(num_epochs, max_num_samples, index_mapping_dir, data_prefix, name, max_seq_length, short_seq_prob, seed):
    ''' This enforces the name index structure to make sure when parameters change, we can still load the same index.
    This is useful for any sample_indexed file.


    This should be used like this:
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
    '''
    import os

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples " "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    if index_mapping_dir is not None:
        indexmap_filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'
    return indexmap_filename

class AltUniref90ClusterMappingDataset(MappedDataset):
    '''
    MappedDataset class with 'create_sample_mapping' implementation. 'create_sample_mapping(dataset)'

    Resulting `sample_mapping` field maps nemo_upsampling_dataset indicies to uniref90_dataset indicies.


    Cluster ID      Cluster Name    Cluster members
    UniRef50_A0A007 Cluster: MoeK5  A0A007
    UniRef50_A0A009F4B0     Cluster: Uncharacterized protein (Fragment)     A0A009F4B0
    UniRef50_A0A009F4U9     Cluster: Bacterial regulatory, tetR family domain protein (Fragment)    A0A009F4U9
    UniRef50_A0A009F6B7     Cluster: Hemagglutinin family protein (Fragment)        A0A009F6B7
    UniRef50_A0A009F6Q0     Cluster: Peptidase C39 family protein (Fragment)        A0A009F6Q0
    UniRef50_A0A009F6T5     Cluster: Concanavalin A-like lectin/glucanases superfamily protein (Fragment)   A0A009F6T5
    UniRef50_A0A009F825     Cluster: Uncharacterized protein        A0A009F825; A0A7X5RNJ4; A0A829RTI8; UPI00112C051E; UPI001580B9E7; UPI00158
    0D784; UPI00070F2CF0; UPI001D187985; UPI00227B0253
    '''
    def __init__(
            self, uniref50_dataset: 'NeMoUpsampling', uniref90_dataset: Dataset, cluster_map_json_path: str,
            data_prefix,
            seed=None,
            index_mapping_dir=None,
            name=None,
            force_regen_sample_mapping=False,
            uniref90_samplemap=None,
            buffer_size=int(1e6),
        ):
        '''
        Creates a mapped dataset mapping samples from uniref50_dataset to cluster members inside uniref90_dataset.
        This happens by first loading the uniref50 dataset, uniref90 dataset, and some cluster mapping taking cluster ids (from uniref50)
        to members. Next we construct a sample mapping by iterating over the entirety of the uniref50_dataset and assigning a random cluster id
        to each member (element in the uniref50_dataset). This is set as the sample mapping used by the underlying MappedDataset.

        cluster_map_json_path - path to a json file encoding the cluster mappings. the keys should correspond _exactly_ to the values
            returned by the first column (data_col=1) of the clustered (uf50) fasta -> processed csv, and the values should similarly correspond _exactly_
            to the values returned by the first column of the member (uf90) fasta -> processed csv

        uniref50_dataset - MUST be an already upsampled dataset on uniref50 that returns cluster_ids (NOT sequences)
            Typically this is done by setting the data_col parameter in the model.data section of the yaml config file.

        uniref90_dataset - just a regular dataset constructed on uniref90, this should return sequences.
        '''

        self.data_prefix = data_prefix
        self.seed = seed
        # Really obnoxious to get these.
        self.index_mapping_dir = index_mapping_dir
        self.name = name
        self.buffer_size = buffer_size

        # Used for creating the cluster_member_id -> index map
        if index_mapping_dir is not None:
            indexmap_filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
        else:
            indexmap_filename = data_prefix

        # At this point we have the directory
        # this gets invoked to create the cluster map, which is used in create_sample_map

        # Cluster map should be passed in some unknown format. Using pre-created JSON for now.
        self.cluster_map_json_path = cluster_map_json_path
        # Pass in the dataset that sample_mapping indicies correspond to
        num_samples=0 # This has no effect on behavior
        logging.info(f"Loading cluster map {self.cluster_map_json_path=}")
        time_start = time.time()
        cluster_map = self.create_cluster_map_from_json(self.cluster_map_json_path)
        time_end = time.time()
        logging.info(f"Cluster map from json: {time_end - time_start}")
        time_start = time.time()
        self.sample_map = self._create_sample_mapping(uniref50_dataset, cluster_map, buffer_size=self.buffer_size)
        time_end = time.time()
        logging.info(f"Cluster sample_mapping: {time_end - time_start}")
        super().__init__(uniref90_dataset, num_samples)
        # Now we can free the dictonaries used for creating the cluster mapping
        del(cluster_map)

    @staticmethod
    def create_cluster_map_from_json(cluster_map_json) -> Dict[str, str]:
        ''' Creates a mapping from cluster_id to cluster_members. This is specifically for mapping samples from
        Uniref50 to Uniref90.

        Json file is expected to be an exact production (meaning, json.loads is sufficient)
        '''
        with open(cluster_map_json, 'r') as fd:
            cluster_map = json.load(fd)

        for k, v in cluster_map.items():
            cluster_map[k] = np.memmap(v)

        return cluster_map

    @staticmethod
    def _create_sample_mapping(uniref50_dataset, cluster_map, buffer_size=int(1e6)):
        """

        Args:
            uniref50_dataset (_type_): _description_
                items are effectively string:
                    ["u50_id0","u50_id1", "u50_id2", ...]
            cluster_map (_type_): _description_
                maps u50 clster to List[u90 cluster]
                 {"u50_id{i}': ["u90_id{j" for j in range(num_maps)]}
            uniref90_samplemap (_type_): _description_
                List[Dict["sequence_id", "..."]]

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            _type_: _description_
        """
        # TODO, in my mind, we don't actually want the cluster_ids, we want their index in
        # the datasets, which for a NeMoUpsampling dataset, should be the sample_mapping
        n_samples = len(uniref50_dataset)

        # sample_map = np.zeros(n_samples) # TODO convert this to memory map
        import tempfile
        tf =  tempfile.NamedTemporaryFile()
        sample_map = np.memmap(tf.name, dtype=int, mode='w+', shape=(n_samples,))

        logging.debug(f"Creating uf50->uf90 sample mapping for {len(uniref50_dataset)} samples.")
        # TODO let's make this buffered
        # TODO I need an index structure for the cluster map so I can tell how many there are
        cluster_counts = cluster_map['counts']
        buffer_start = 0
        while buffer_start < n_samples:
            buffer_end = buffer_start + buffer_size
            cluster_slice = uniref50_dataset.sample_map[buffer_start:buffer_end]
            sample_slice = np.random.rand(len(cluster_slice))
            cluster_count_slice = cluster_counts[cluster_slice]
            sampled_relative_indices = np.rint(sample_slice * cluster_count_slice)
            sample_map[buffer_start:buffer_end] = cluster_map['starts'][cluster_slice] + sampled_relative_indices
            buffer_start = buffer_end

        return sample_map

    def create_sample_mapping(self, dataset, num_samples=None) -> np.array:
        '''
        Creates a sample mapping from our current (uf50) dataset to an index in our new dataset (uf90) by choosing members
            from a cluster map. this mimicks the behavior described in the ESM2 publication.
        dataset - dataset that sample_mapping indexes into.
        num_samples - This is an unused parameter.

        Returns - numpy array that maps indicies from the first dataset (Uniref50) to an entry in the second dataset (Uniref90)
        '''
        return self.sample_map

class Uniref90ClusterMappingDataset(MappedDataset):
    '''
    MappedDataset class with 'create_sample_mapping' implementation. 'create_sample_mapping(dataset)'

    Resulting `sample_mapping` field maps nemo_upsampling_dataset indicies to uniref90_dataset indicies.


    Cluster ID      Cluster Name    Cluster members
    UniRef50_A0A007 Cluster: MoeK5  A0A007
    UniRef50_A0A009F4B0     Cluster: Uncharacterized protein (Fragment)     A0A009F4B0
    UniRef50_A0A009F4U9     Cluster: Bacterial regulatory, tetR family domain protein (Fragment)    A0A009F4U9
    UniRef50_A0A009F6B7     Cluster: Hemagglutinin family protein (Fragment)        A0A009F6B7
    UniRef50_A0A009F6Q0     Cluster: Peptidase C39 family protein (Fragment)        A0A009F6Q0
    UniRef50_A0A009F6T5     Cluster: Concanavalin A-like lectin/glucanases superfamily protein (Fragment)   A0A009F6T5
    UniRef50_A0A009F825     Cluster: Uncharacterized protein        A0A009F825; A0A7X5RNJ4; A0A829RTI8; UPI00112C051E; UPI001580B9E7; UPI00158
    0D784; UPI00070F2CF0; UPI001D187985; UPI00227B0253
    '''
    def __init__(
            self, uniref50_dataset: 'NeMoUpsampling', uniref90_dataset: Dataset, cluster_map_json_path: str,
            data_prefix,
            seed=None,
            index_mapping_dir=None,
            name=None,
            force_regen_sample_mapping=False,
            uniref90_samplemap=None,
        ):
        '''
        Creates a mapped dataset mapping samples from uniref50_dataset to cluster members inside uniref90_dataset.
        This happens by first loading the uniref50 dataset, uniref90 dataset, and some cluster mapping taking cluster ids (from uniref50)
        to members. Next we construct a sample mapping by iterating over the entirety of the uniref50_dataset and assigning a random cluster id
        to each member (element in the uniref50_dataset). This is set as the sample mapping used by the underlying MappedDataset.

        cluster_map_json_path - path to a json file encoding the cluster mappings. the keys should correspond _exactly_ to the values
            returned by the first column (data_col=1) of the clustered (uf50) fasta -> processed csv, and the values should similarly correspond _exactly_
            to the values returned by the first column of the member (uf90) fasta -> processed csv

        uniref50_dataset - MUST be an already upsampled dataset on uniref50 that returns cluster_ids (NOT sequences)
            Typically this is done by setting the data_col parameter in the model.data section of the yaml config file.

        uniref90_dataset - just a regular dataset constructed on uniref90, this should return sequences.
        '''

        self.data_prefix = data_prefix
        self.seed = seed
        # Really obnoxious to get these.
        self.index_mapping_dir = index_mapping_dir
        self.name = name

        # Used for creating the cluster_member_id -> index map
        if index_mapping_dir is not None:
            indexmap_filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
        else:
            indexmap_filename = data_prefix

        # At this point we have the directory
        # this gets invoked to create the cluster map, which is used in create_sample_map

        # Cluster map should be passed in some unknown format. Using pre-created JSON for now.
        self.cluster_map_json_path = cluster_map_json_path
        # Pass in the dataset that sample_mapping indicies correspond to
        # NOTE: Do we needthis as an attribute?
        # self.uniref50_dataset = uniref50_dataset

        if uniref90_samplemap is None:
            # Convenience method incase the caller doesnt know what is going on, but still wants to run.
            self.sample_map_json_fn= indexmap_filename
            self.sample_map_json_fn += f'seed{seed}_sample_mapping_cache.json'
            # Basically we do not want this as an attribute, ever. how do we avoid that?
            logging.info(f"Creating sample mapping cache {self.cluster_map_json_path}")
            time_start = time.time()
            uniref90_samplemap = self._create_sample_mapping_cache(self.sample_map_json_fn, uniref90_dataset, force=force_regen_sample_mapping)
            time_end = time.time()
            logging.info(f'Sample mapping cache construction: {time_end - time_start}')

        num_samples=0 # This has no effect on behavior
        logging.info(f"Loading cluster map {self.cluster_map_json_path=}")
        time_start = time.time()
        cluster_map = self.create_cluster_map_from_json(self.cluster_map_json_path)
        time_end = time.time()
        logging.info(f"Cluster map from json: {time_end - time_start}")
        time_start = time.time()
        self.sample_map = self._create_sample_mapping(uniref50_dataset, cluster_map, uniref90_samplemap)
        time_end = time.time()
        logging.info(f"Cluster sample_mapping: {time_end - time_start}")
        super().__init__(uniref90_dataset, num_samples)
        # Now we can free the dictonaries used for creating the cluster mapping
        del(uniref90_samplemap)
        del(cluster_map)

    @staticmethod
    def _create_sample_mapping_cache(filename, uniref90_dataset, force=False) -> Dict[str, int]:
        ''' Creates a map from cluster member id (key) to position in a dataset (value).

        caches the result as a json file.
        '''
        # This can happen when the dataset is ready, e.g. part of the constructor
        if os.path.exists(filename) and not force:
            logging.debug(f'restoring found sample index map {filename}')
            with open(filename, 'r') as fd:
                uniref90_samplemap = json.load(fd)
            logging.debug(f'Done')
            # Now we do it again and check that it still works. (seems like it does), unirefid-ot-int-2.json for context
        else:
            logging.debug(f"building sample index map for {len(uniref90_dataset)} samples")
            # this is a map that does sample_id to index

            # TODO: DataLoader here to go fast?
            uniref90_samplemap = {k['sequence_id']: i for i, k in enumerate(uniref90_dataset)}
            with open(filename, 'w') as fd:
                json.dump(uniref90_samplemap, fd)
        return uniref90_samplemap

    @staticmethod
    def create_cluster_map_from_json(cluster_map_json) -> Dict[str, str]:
        ''' Creates a mapping from cluster_id to cluster_members. This is specifically for mapping samples from
        Uniref50 to Uniref90.

        Json file is expected to be an exact production (meaning, json.loads is sufficient)
        '''
        with open(cluster_map_json, 'r') as fd:
            cluster_map = json.load(fd)
        return cluster_map

    @staticmethod
    def _create_sample_mapping(uniref50_dataset, cluster_map, uniref90_samplemap):
        sample_map = list()
        logging.debug(f"Creating uf50->uf90 sample mapping for {len(uniref50_dataset)} samples.")
        for a in tqdm(uniref50_dataset):
            if a in cluster_map:
                items = cluster_map[a]
                match items:
                    case [item]:
                        selected_sample_id = item
                    case _:
                        selected_sample_id = random.choice(items)
                # selected_sample_id = random.choice(self.cluster_map[a])
            else:
                # These are probably from bad versions
                logging.warning(f"Missing entry in clustermap {a}, this breaks sampling assumptions")
                # We cant continue here because it will break things, particularly with our index.
                raise Exception(f"Cannot create a sample mapping for cluster: {a} with no corresponding cluster members. Check cluster mapping.")

            # Get the actual index, add to our sample map
            if (selected_sample_idx := uniref90_samplemap.get(selected_sample_id)) is not None:
                # sample_map.append(selected_sample_idx)
                sample_map.append(selected_sample_idx)
            else:
                # Occurs when the dictionary going from member => idx has no entry for some cluster member.
                #   Causes:
                #       - cluster map contains members that are not present in the underlying dataset
                #       - cluster map contains non-existant members
                raise Exception("Cluster member id to index map is corrupted.")
        return np.array(sample_map)

    def create_sample_mapping(self, dataset, num_samples=None) -> np.array:
        '''
        Creates a sample mapping from our current (uf50) dataset to an index in our new dataset (uf90) by choosing members
            from a cluster map. this mimicks the behavior described in the ESM2 publication.
        dataset - dataset that sample_mapping indexes into.
        num_samples - This is an unused parameter.

        Returns - numpy array that maps indicies from the first dataset (Uniref50) to an entry in the second dataset (Uniref90)
        '''
        return self.sample_map

class NeMoUpsampling(MappedDataset):
    """Upsamples a dataset to a target length by repeating samples."""
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
        logging.debug(f"Filtered out (ignored) {ds_size - filtered_ds_size} samples ( {filtered_ds_size} / {ds_size} )")

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
