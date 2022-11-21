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
from nemo.utils import logging
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import (
    TextMemMapDataset
)
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
    >>> dataset = FastaDataset(fasta, max_length=4)
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


class FastaMemMapDataset(TextMemMapDataset):
    """
    See FASTA format definition here:
    http://www.ncbi.nlm.nih.gov/blast/fasta.shtml
    E.g., the following is a valid FASTA
    >seq1
    ACGTAC
    GAGATA
    >seq2
    TACATA
    A useful access would be, e.g.,
    >>> dataset = FastaMemMapDataset(files, max_length=4)
    >>> dataset[0]
    'ACGT'
    >>> dataset[4]
    'ACGA'
    >>> dataset[9]
    'TACA'
    >>> dataset[11]
    'CATA'
    The general pattern for getting substrings from the FASTA is:
    On initialization:
    1. Infer the lines to ignore (i.e., description lines)
    2. Build an index of new line characters (this is saved, use the stored
        index if it exists)
    3. Build an index of the DNA sequences specifically based
        on the buffer size of this reader (i.e., `max_length`).
    On get:
    4. Use indexes to figure out the portion of string to pull into memory
    5. Post-process to strip new lines
    This could also be applicable to sequences from FASTQ, but is complicated
        by files including a couple extra lines for quality scores.
    This parser will need a little bit of additional sophistication to return
        the description and relative position of a sequence.
    """
    def __init__(
        self,
        dataset_paths, max_length, newline_int=10, header_lines=0,
        workers=None, tokenizer=None, sort_dataset_paths=True,
        escape_char=62,
    ):
        """
        Args:
            dataset_paths (List[str]): FASTA files to be included
            max_length (int): Length of sequences to load. Sequences will be
                less than `max_length` if they are in the last `max_length` - 1
                bases in the entry.
            newline_int (Optional[int]): The integer to parse as newline
                characters in the memory map.
            header_lines (Optional[int]): Number of lines to skip at beginning
                of file
            workers (Optional[int]): Number of works used to build the file
                indexes.
            tokenizer (Optional[TokenizerSpec]): If supplied, the string will
                be tokenized by `tokenizer` before being returned.
            sort_dataset_paths (Optional[bool]): If True, the files are sorted
                in lexicographic order for accession.
            escape_char (Optional[int]): ignore lines that start with
                `escape_char`. Note: 62 is the result of
                int.from_bytes('>'.encode('utf-8'), "big")

        """
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=newline_int,
            header_lines=header_lines,
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
        )
        self.max_length = max_length
        self.escape_char = escape_char

        startline_index_list = []
        for _, midx in self.mdata_midx_list:
            startlines = np.array([0, *midx[:-1]])
            startline_index_list.append(startlines)

        self.sequence_index_bins = []
        self.sequence_indexes = []

        self._build_sequence_indices()


    def _build_sequence_indices(self):
        """
        Builds additional indices for sequence specific indexing.
        There is one index to keep track of the number of indexable positions
        in each file.
        Then there is an index of cumulative number of indexable sequence bases
        ending on each line of the file.
        """
        self.length_bins = [0]
        for mdata, midx in self.mdata_midx_list:
            sequence_index = self._build_sequence_index(
                    mdata, midx, self.escape_char,
                )
            self.length_bins.extend(
                    self._extract_sequence_lengths(sequence_index)
                )
            self.sequence_indexes.append(sequence_index)
            if len(self.sequence_index_bins) > 0:
                new_bin = sequence_index[-1] + self.sequence_index_bins[-1]
            else:
                new_bin = sequence_index[-1]

            self.sequence_index_bins.append(new_bin)
        self.length_bins = np.array(self.length_bins)

    def _extract_sequence_lengths(self, sequence_index):
        length_bins = []
        last_seq_len = 0
        for i in range(len(sequence_index) - 1):
            if sequence_index[i] == sequence_index[i + 1]:
                self.length_bins.append(sequence_index[i])
                last_seq_len = sequence_index[i] - last_seq_len
        length_bins.append(sequence_index[-1])
        return length_bins

    @staticmethod
    def _build_sequence_index(mdata, midx, escape_char):
        """
        Create an index of the amount of sequence bases that have ocurred by
        the end of each line. Sequence descriptions are ignored.
        E.g., for the following FASTA:
        ```
        >seq1
        ACAGAT
        TCGACCC
        >seq2
        TACAT
        ```
        The result would be: [0, 6, 13, 13, 18]
        """
        # Note: We are creating a lot of arrays here, these have the potential
        #  to be pretty huge on really large files. See if there is a way to
        #  reduce memory usage. We can also store extra indices in an npy file.
        start_positions = np.concatenate(([0], midx[:-1] + 1))
        line_lengths = midx - start_positions
        start_chars = mdata[start_positions]
        is_kept_mask = start_chars != escape_char
        masked_line_lengths = line_lengths * is_kept_mask

        return np.cumsum(masked_line_lengths)

    def __len__(self):
        """The number of positions that can be sampled from the dataset"""
        return self.sequence_index_bins[-1]

    def __getitem__(self, idx):
        """
        Return a string from binary memmap
        """

        idx = handle_index(self, idx)

        record, start, end = self._get_record_start_end(idx, self.max_length)

        text = record[start:end].tobytes().decode("utf-8")

        # parse raw text (e.g., tokenize)
        data = self._build_data_from_text(text)

        return data

    def _build_data_from_text(self, text: str):
        text = text.replace('\n', '').split('>')[0]
        text = super()._build_data_from_text(text)
        data = dict(seq=text)
        return data

    def _get_record_start_end(self, idx, length):
        """
        Retrieve the start and end of the index requested.
        Since sequences in FASTA can have arbitrary new lines in the middle
        of them, this function does additional work to ensure that the correct
        number of non-newline characters are returned.
        The extra work is linear with the number of newline characters, so
        files can be pre-processed to remove the newlines to speed up access,
        but this indexing will work in either case.
        """
        # Identify the file containing the record
        file_id = np.digitize(idx, self.sequence_index_bins, right=False)
        base_idx = self.sequence_index_bins[file_id - 1] if file_id > 0 else 0
        within_file_idx = idx - base_idx + self._header_lines
        mdata, midx = self.mdata_midx_list[file_id]
        seq_index = self.sequence_indexes[file_id]

        start_line = np.digitize(within_file_idx, seq_index)
        # the condition where start_line == 0 should not be invoked for real
        #  FASTA files, but this function will work regardless
        position_at_start_of_line = \
            (midx[start_line - 1] + 1) if start_line else 0

        start_position_within_line = (within_file_idx - \
            seq_index[start_line - 1]) if start_line else within_file_idx

        start_index = position_at_start_of_line + start_position_within_line

        # calculate number of new lines
        current_pos = start_index
        current_line = start_line
        accumualted_chars = 0
        number_of_new_line_chars = 0

        # The length of the string here can vary because a desired string from
        # a fasta may be split over an arbitrary number of lines.
        while accumualted_chars < length and current_line < len(midx):
            # accumulate the end of this line, or as much as possible
            chars_remaining = length - accumualted_chars
            chars_on_line = midx[current_line] - current_pos
            accumualted_chars += min(
                chars_remaining, chars_on_line
            )
            if chars_remaining > chars_on_line:
                number_of_new_line_chars += 1
            current_pos += chars_remaining + 1
            current_line += 1

        end_index = start_index + length + \
            number_of_new_line_chars
        return mdata, start_index, end_index


class DiscretizeFastaDataset(MappedDataset):

    def __init__(self, dataset: FastaDataset):
        """
        Produces a discretized version of a `FastaDataset`.

        `DiscretizeFastaDataset` wraps a `FastaDatset` to produce non-overlapping
        subsections, e.g.: a dataset built on `ACGT` with sequence length of 2
        will only yield `AC` and `GT`. This is in contrast to the
        `FastaDataset`, which provides overlapping sections of the sequence,
        e.g., `ACGT` with a sequence length of 2 will return `AC`, `CG`, `GT`,
        and `T`.

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

def pyfastx_constructor(builder, discretize, max_length):
    transforms = builder.make_transforms(discretize)
    return ConcatFastaDataset(
            builder.dataset_paths, max_length,
            backend=builder.options['dataset_backend'],
            transforms=transforms,
        )

def fasta_memmap_constructor(builder, discretize, max_length):
    if discretize:
        datasets = [DiscretizeFastaDataset(FastaMemMapDataset(ds, max_length))
            for ds in builder.dataset_paths]
        dataset = ConcatDataset(datasets)
    else:
        dataset = FastaMemMapDataset(
                builder.dataset_paths, max_length,
            )
    return dataset

tokenizers = {
    'kmer': KmerTokenizer,
}

adapters = {
    'kmer': KmerTokenizerAdapter,
}

dataset_constructors = {
    'pyfastx': pyfastx_constructor,
    'memmap': fasta_memmap_constructor
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
        dataset_class = self.options['dataset_class']
        max_length = cfg.seq_length - 1 + cfg.k
        constructor = dataset_constructors[dataset_class]
        self.dataset = constructor(self, discretize=discretize, max_length=max_length)
        return self.dataset


    def make_transforms(self, discretize):
        """
        Makes transformations to use for the a Dataset.

        Arguments:
            discretize (bool): whether the Discretize argument should be added
        Returns:
            List[Callable[[Dataset], Dataset]]: Dataset transformations

        """
        transforms = [DiscretizeFastaDataset,] if discretize else []
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
            dataset_format = cfg.get('dataset_format')
            dataset_class = cfg.get('dataset_class', 'memmap')
            dataset_backend = cfg.get('dataset_backend')
            if dataset_class == 'pyfastx':
                if dataset_backend not in BACKENDS.keys():
                    raise ValueError(
                        f'dataset_backend={dataset_backend} for dataset_class='
                        f'{dataset_class} must be configured to one of: '
                        f'{BACKENDS.keys()}')

        # Build individual datasets.
        filepath = os.path.join(dataset_path, name, ds)

        options = {
            'cfg': cfg,
            'filepath': filepath,
            'dataset_format': dataset_format,
            'batch_size': self.get_global_batch_size(),
            'dataset_class': dataset_class,
            'dataset_backend': dataset_backend,
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
        options['discretize'] = self.cfg.discretize_train

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
