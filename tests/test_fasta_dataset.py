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

import gzip
import pytest
import tempfile
import numpy as np
import numpy.testing as npt
from bionemo.data.fasta_dataset import (
    FastaDataset, ConcatFastaDataset,
    BACKENDS,
)
import pyfastx

def test_fastx_dataset_check_len():
    fasta_file = tempfile.NamedTemporaryFile()
    with open(fasta_file.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGACCC\n>seq2\nTACAT\n')

    fa = pyfastx.Fasta(fasta_file.name)
    dataset = FastaDataset(fa, 4)

    assert len(dataset) == 18

test_examples = [
    (-1, 'T'),
    (-2, 'AT'),
    (-5, 'TACA'),
    (-6, 'C'),
    (-10, 'GACC'),
    (0, 'ACAG'),
    (1, 'CAGA'),
    (2, 'AGAT'),
    (3, 'GATT'),
    (4, 'ATTC'),
    (10, 'CCC'),
    (11, 'CC'),
    (12, 'C'),
    (13, 'TACA'),
    (-18, 'ACAG'),
]
@pytest.mark.parametrize('backend', BACKENDS.keys())
@pytest.mark.parametrize('idx,sequence', test_examples)
def test_fasta_dataset(idx, sequence, backend):
    fasta_file = tempfile.NamedTemporaryFile()
    with open(fasta_file.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGACCC\n>seq2\nTACAT\n')

    fa = pyfastx.Fasta(fasta_file.name)
    dataset = FastaDataset(fa, 4, backend=backend)
    assert dataset[idx]['seq'] == sequence

test_examples_multiple_files = [
    (0, 'ACAG', 'seq1', 0),
    (3, 'GATT', 'seq1', 3),
    (13, 'TACA', 'seq2', 0),
    (17, 'T', 'seq2', 4),
    (18, 'CCCA', 'seq1:2', 0),
    (30, 'ACAT', 'seq2:2', 1),
]
@pytest.mark.parametrize('backend', BACKENDS.keys())
@pytest.mark.parametrize(
        'idx,sequence,contig,start',
        test_examples_multiple_files
    )
def test_fasta_memmap_dataset_multiple_files(
        idx,
        sequence,
        contig,
        start,
        backend):
    fasta_file_1 = tempfile.NamedTemporaryFile()
    with open(fasta_file_1.name, 'w') as fh:
        fh.write('>seq1\nACAGAT\nTCGACCC\n>seq2\nTACAT\n')

    fasta_file_2 = tempfile.NamedTemporaryFile()
    with gzip.open(fasta_file_2.name, 'wb') as fh:
        fh.write(b'>seq1:2\nCCCATTNA\nNAT\n>seq2:2\nTACATAC\nATATTC\n')

    dataset = ConcatFastaDataset(
        [fasta_file_1.name, fasta_file_2.name],
        max_length=4,
        backend=backend,
        )
    assert dataset[idx]['seq'] == sequence
    assert dataset[idx]['contig'] == contig
    assert dataset[idx]['start'] == start
