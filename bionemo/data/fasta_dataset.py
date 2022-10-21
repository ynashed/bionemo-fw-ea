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

import os
from copy import deepcopy
import pyfastx
from nemo.core import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
from omegaconf import open_dict
from bionemo.data.utils import (
    MappedDataset,
    handle_index,
)
from bionemo.data.dataloader.kmer_collate import DeterministicLengthTruncator
from bionemo.tokenizer import KmerTokenizer
from bionemo.data.dataloader import (
    KmerBertCollate,
    SpanMasking,
    LengthTruncator,
    KmerTokenizerAdapter,
)

from bionemo.data.utils import (
    FormattedDatasetFactory,
    DatasetBuilderSpec,
    expand_dataset_paths,
)
from bionemo.data.utils import NeMoUpsampling
from bionemo.core import BioNeMoDataModule


class _InMemoryFastxBackend:
    def __init__(self):
        self.records = dict()

    def write(self, fastx):
        for seq in fastx:
            self.records[seq.name] = seq.seq

    def fetch(self, contig, start, end):
        return self.records[contig][start:end]


class _FetchFastxBackend:

    def __init__(self):
        self.fastx = None

    def write(self, fastx):
        self.fastx = fastx

    def fetch(self, contig, start, end):
        return self.fastx.fetch(contig, (start + 1, end))

BACKENDS = {
    'file': _FetchFastxBackend,
    'memory': _InMemoryFastxBackend,
}


class FastaDataset(Dataset):
    """
    Constructs a dataset for a pyfastx.Fasta

    See FASTA format definition here:
    http://www.ncbi.nlm.nih.gov/blast/fasta.shtml

    E.g., the following is a valid FASTA
    >seq1
    ACGTAC
    GAGATA
    >seq2
    TACATA

    A useful access would be, e.g.,
    >>> fasta = pyfasx.Fasta('example.fa')
    >>> dataset = FastaDataset(fasta, seq_length=4)
    >>> dataset[0]
    'ACGT'
    >>> dataset[4]
    'ACGA'
    >>> dataset[9]
    'ATA'
    >>> dataset[13]
    'ACAT'

    """

    def __init__(self, fastx_file, max_length, backend='file'):
        """

        Arguments:
            fastx_file (pyfastx.Fasta): Fasta object for the dataset.
            max_length (int): Maximum length of sub-sequences to fetch from the
                FASTA.
            backend (Literal['file', 'memory']): If 'file',
                the data is read from file on the fly with `pyfastx`. If
                'memory', the data is stored in memory.

        """
        self.fastx = fastx_file
        self.max_length = max_length
        self.backend = BACKENDS[backend]()
        self.backend.write(self.fastx)
        bins = [(seq.name, len(seq)) for seq in self.fastx]
        self.name_bins = [bin[0] for bin in bins]
        self.length_bins = np.cumsum([bin[1] for bin in bins])

    def fetch(self, id_, start, end):
        return self.backend.fetch(id_, start, end)

    def ids(self):
        return list(self.fastx.keys())

    def __len__(self):
        """Return the length of the dataset."""
        return self.length_bins[-1]

    def __getitem__(self, idx):
        """Return the subsequence starting at idx

        Returns:
            Dict: containing the sequence, the contig/description,
            and the position at the index.
        """
        idx = handle_index(self, idx)

        contig_idx = min(
            np.digitize(idx, self.length_bins),
            len(self.length_bins) - 1
            )
        contig_offset = self.length_bins[contig_idx - 1] \
            if contig_idx > 0 else 0
        contig_name = self.name_bins[contig_idx]
        # wrap in int because floats here will cause errors in `fetch`
        position_start = int(idx - contig_offset)
        position_end = int(position_start + self.max_length)
        text = self.backend.fetch(
            contig_name,
            position_start, position_end
        )
        return {
            'seq': text,
            'contig': contig_name,
            'start': position_start,
        }


class Discretize(MappedDataset):

    def __init__(self, dataset: FastaDataset):
        """
        Produces a discretized version of a `FastaDataset`.

        The `__getitem__` call of `FastaDataset` can provide overlapping
        sections of the sequence, e.g., `ACGT` with a sequence length of 2
        can return `AC`, `CG`, `GT`, and `T`. `Discretize` wraps the
        `FastaDatset` to produce non-overlapping subsections, e.g.: a dataset
        built on `ACGT` with sequence length of 2 will only yield `AC` and
        `GT`.

        Arguments:
            dataset (FastaDataset): a dataset to discretize. The chunk size
                used to discretize the dataset will be inferred from `dataset`.

        """
        super().__init__(dataset, None)

    def fetch(self, id_, start, end):
        return self._dataset.fetch(id_, start, end)

    def ids(self):
        return self._dataset.ids()

    @staticmethod
    def create_sample_mapping(dataset, num_samples):
        """Sample mapping used to remap a dataset into discrete sections.

        Arguments:
            dataset (FastaDataset): dataset to discretize
            num_samples (None): Unused.

        Returns:
            sample_mapping (ArrayLike[int]): If `sample_mapping[i] == j`,
            the `i`th entry in this dataset will be `j`th entry of the original
            dataset.

        """
        first_bin = dataset.length_bins[0]
        rest_bin =  (dataset.length_bins[1:] - dataset.length_bins[:-1])
        bins = np.append(first_bin, rest_bin)
        # shortcut for ceil division
        n_seqs_per_bin = -(bins // -dataset.max_length)
        sample_mapping = np.zeros(n_seqs_per_bin.sum(), dtype=int)
        pos = 0
        for i, n_seqs in enumerate(n_seqs_per_bin):
            initial_start = 0 if i == 0 else dataset.length_bins[i - 1]
            for j in range(n_seqs):
                start = initial_start + j * dataset.max_length
                sample_mapping[pos] = start
                pos += 1
        return sample_mapping


class ConcatFastaDataset(Dataset):
    def __init__(self, files, max_length, backend='file', uppercase=False,
                 transforms=None,
                 ):
        """
        Constructs a dataset consisting of multiple FASTA files.

        Arguments:
            files (List[str]): Each entry contains a filepath to a FASTA file.
                The files can be zipped (gzip) or not zipped. An arbtrary
                mixture of both zipped and not zipped is supported.
            max_length (int): Maximum length of sub-sequences to fetch from the
                FASTA.
            backend (Literal['file', 'memory']): Either 'file' or 'memory'. If 'file',
                the data is read from file on the fly with `pyfastx`. If
                'memory', the data is stored in memory.
            uppercase (Optional[bool]). If True, the sequences are coerced to
                uppercase. Otherwise, they are provided in the original case
                from the file.
            transforms (Optional[List[Callable[[Dataset], Dataset]]): If
                specified, each entry of `transforms` returns a dataset. The
                entries will be applied sequentially to the components of this
                dataset.

        """
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.datasets = []
        for f in files:
            new_dataset = FastaDataset(
                pyfastx.Fasta(f, uppercase=uppercase),
                max_length, backend=backend,
            )
            self.datasets.append(self._apply_transforms(new_dataset))
        self._dataset = ConcatDataset(self.datasets)
        self.ids = {}
        self.build_id_index()

    def build_id_index(self):
        for ds in self.datasets:
            if hasattr(ds, 'ids'):
                for key in ds.ids():
                    # TODO warn for overriding key?
                    self.ids[key] = ds

    def fetch(self, id_, start, end):
        ds = self.ids[id_]
        return ds.fetch(id_, start, end)

    def _apply_transforms(self, dataset):
        for transform in self.transforms:
            dataset = transform(dataset)
        return dataset

    def __len__(self):
        """Return the length of the dataset."""
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return the subsequence starting at idx

        Returns:
            Dict: containing the sequence, the contig/description,
            and the position at the index.
        """
        return self._dataset[idx]


tokenizers = {
    'kmer': KmerTokenizer,
}

adapters = {
    'kmer': KmerTokenizerAdapter,
}


class FastaDatasetBuilder(DatasetBuilderSpec):

    def format_dataset_paths(self):
        """
        Parses FASTA paths.

        """
        self.dataset_paths = expand_dataset_paths(
            self.options['filepath'], None)

    def check_path(self, filepath):
        """
        Checks whether a FASTA exists.

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
        Instantiates a FastaDataset.

        Returns:
            Dataset: Dataset instantiated from paths.
        """
        cfg = self.options['cfg']
        discretize = self.options['discretize']
        max_length = cfg.seq_length - 1 + cfg.k
        transforms = self.make_transforms(discretize)
        self.dataset = ConcatFastaDataset(
            self.dataset_paths, max_length, backend='memory',
            transforms=transforms,
        )
        return self.dataset

    def make_transforms(self, discretize):
        """
        Makes transformations to use for the a Dataset.

        Arguments:
            discretize (bool): whether the Discretize argument should be added
        Returns:
            List[Callable[[Dataset], Dataset]]: Dataset transformations

        """
        transforms = [Discretize,] if discretize else []
        return transforms


class DNABERTDatasetFactory(FormattedDatasetFactory):

    def __init__(self):
        """
        Initializes a dataset factory for handling fasta formats.
        """
        self.formats = {
            'fasta': FastaDatasetBuilder,
        }


class DNABERTDatasetFactory(FormattedDatasetFactory):

    def __init__(self):
        """
        Initializes a dataset factory for handling fasta formats.
        """
        self.formats = {
            'fasta': FastaDatasetBuilder,
        }

tokenizers = {
    'kmer': KmerTokenizer,
}

adapters = {
    'kmer': KmerTokenizerAdapter,
}


class FastaDatasetBuilder(DatasetBuilderSpec):

    def format_dataset_paths(self):
        """
        Parses FASTA paths.

        """
        self.dataset_paths = expand_dataset_paths(
            self.options['filepath'], None)

    def check_path(self, filepath):
        """
        Checks whether a FASTA exists.

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
        Instantiates a FastaDataset.

        Returns:
            Dataset: Dataset instantiated from paths.
        """
        cfg = self.options['cfg']
        discretize = self.options['discretize']
        max_length = cfg.seq_length - 1 + cfg.k
        transforms = self.make_transforms(discretize)
        self.dataset = ConcatFastaDataset(
            self.dataset_paths, max_length, backend='memory',
            transforms=transforms,
        )
        return self.dataset

    def make_transforms(self, discretize):
        """
        Makes transformations to use for the a Dataset.

        Arguments:
            discretize (bool): whether the Discretize argument should be added
        Returns:
            List[Callable[[Dataset], Dataset]]: Dataset transformations

        """
        transforms = [Discretize,] if discretize else []
        return transforms


class DNABERTDataModule(BioNeMoDataModule):

    def __init__(self, cfg, trainer):
        """Initializes a DNABERTDataModule

        Arguments:
            cfg (OmegaConf): A config object for a model
            trainer (pytorch_lightning.Trainer): Trainer of the corresponding
                model.

        """
        super().__init__(cfg, trainer)
        self.init_num_samples()
        self.pad_size_divisible_by_8 = True if self.model_cfg.masked_softmax_fusion else False

    # this can probably be shared by some BioNeMo models that already use the
    # dataset factory or should be refactored from `build_train_valid_test_datasets`
    def _configure_options(self, name, ds):
        cfg = deepcopy(self.cfg)
        with open_dict(cfg):
            dataset_path = cfg.get('dataset_path', '')
            metadata_file = cfg.get('metadata_file', None)
            dataset_format = cfg.get('dataset_format')

        # Build individual datasets.
        filepath = os.path.join(dataset_path, name, ds)
        metadata_path = os.path.join(dataset_path, name, metadata_file) if metadata_file else None

        cfg = deepcopy(cfg)
        with open_dict(cfg):
            cfg['metadata_path'] = metadata_path

        options = {
            'cfg': cfg,
            'filepath': filepath,
            'metadata_path': metadata_path,
            'dataset_format': dataset_format,
            'batch_size': self.get_global_batch_size(),
        }
        return options

    @staticmethod
    def _get_random_length_truncator():
        sentence_transform = LengthTruncator()
        sentence_transform.get_sentence = lambda x: x['seq']
        return sentence_transform

    @staticmethod
    def _get_deterministic_length_truncator():
        sentence_transform = DeterministicLengthTruncator()
        sentence_transform.get_sentence = lambda x: x['seq']
        return sentence_transform

    def _setup_collate(self, model, dataloader, sentence_transform):
        dataloader.collate_fn = KmerBertCollate(
            model.tokenizer,
            seq_length=self.cfg.seq_length,
            pad_size_divisible_by_8=self.pad_size_divisible_by_8,
            masking_strategy=SpanMasking(
                tokenizer=model.tokenizer,
                seed_probability=0.15,
                span_length=model._cfg.tokenizer.k,
            ),
            transform_sentences=sentence_transform,
        ).collate_fn

    def _setup_eval_dataloader(self, model, dataloader):
        sentence_transform = self._get_deterministic_length_truncator()
        self._setup_collate(model, dataloader, sentence_transform)

    def train_dataset(self):
        """Creates a Training dataset for DNABERT

        Returns:
            Dataset: A ConcatFastaDataset.

        """
        ds = self.cfg.dataset.train
        name = 'train'
        options = self._configure_options(name, ds)
        options['discretize'] = False

        dataset = DNABERTDatasetFactory().create_dataset(options)

        return dataset

    def sample_train_dataset(self, dataset):
        """Creates an upsampled version of the DNABERT training dataset

        """
        num_samples = self.train_num_samples
        dataset_dir = os.path.join(self.cfg.dataset_path, 'train',)
        dataset = NeMoUpsampling(
            dataset, num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.dataset.train,
            index_mapping_dir=dataset_dir,
            name='train',
            )
        return dataset

    def val_dataset(self):
        """
        Produces a discretized FASTA dataset to use for validation.
        """
        ds = self.cfg.dataset.val
        name = 'val'
        options = self._configure_options(name, ds)
        options['discretize'] = True
        options['name'] = 'val'
        dataset = DNABERTDatasetFactory().create_dataset(options)
        return dataset

    def sample_val_dataset(self, dataset):
        num_samples = self.val_num_samples
        dataset_dir = os.path.join(self.cfg.dataset_path, 'val',)
        dataset = NeMoUpsampling(
            dataset, num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.dataset.val,
            index_mapping_dir=dataset_dir,
            name='val',
            )
        return dataset

    def test_dataset(self):
        """
        Produces a discretized FASTA dataset to use for testing.
        """
        ds = self.cfg.dataset.test
        name = 'test'
        options = self._configure_options(name, ds)
        options['discretize'] = True
        return DNABERTDatasetFactory().create_dataset(options)

    def adjust_train_dataloader(self, model, dataloader):
        sentence_transform = self._get_random_length_truncator()
        return self._setup_collate(model, dataloader, sentence_transform)

    def adjust_val_dataloader(self, model, dataloader):
        return self._setup_eval_dataloader(model, dataloader)

    def adjust_test_dataloader(self, model, dataloader):
        return self._setup_eval_dataloader(model, dataloader)
