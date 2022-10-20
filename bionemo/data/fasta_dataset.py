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

import pyfastx
from nemo.core import Dataset
from torch.utils.data import ConcatDataset
import numpy as np


def _handle_index(dataset, idx):
    if idx < 0 and idx > -len(dataset) - 1:
        idx = len(dataset) + idx
    elif idx < 0:
        raise IndexError('index out of range')
    return idx


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

    def __len__(self):
        """Return the length of the dataset."""
        return self.length_bins[-1]

    def __getitem__(self, idx):
        """Return the subsequence starting at idx

        Returns:
            Dict: containing the sequence, the contig/description,
            and the position at the index.
        """
        idx = _handle_index(self, idx)

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


class ConcatFastaDataset(Dataset):
    def __init__(self, files, max_length, backend='file', uppercase=False):
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

        """
        self.datasets = []
        for f in files:
            new_dataset = FastaDataset(
                pyfastx.Fasta(f, uppercase=uppercase),
                max_length, backend=backend,
            )
            self.datasets.append(new_dataset)
        self._dataset = ConcatDataset(self.datasets)

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
