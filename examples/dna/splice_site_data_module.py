from bionemo.data.dataloader.kmer_collate import KmerBertCollate
from bionemo.data.fasta_dataset import ConcatFastaDataset, FastaDataset
from bionemo.data.validation_dataset import DataFrameTransformDataset
from bionemo.core import BioNeMoDataModule
import pandas as pd
from functools import partial
from bionemo.data.utils import expand_dataset_paths
from bionemo.data.utils import NeMoUpsampling
import os
import math

import numpy as np

def get_start_end(coord, length):
    start = int(coord - math.ceil(length / 2))
    end = int(coord + math.floor(length / 2))
    return start, end

def fetch_bert_dna(row: pd.Series, dataset: FastaDataset, bert_prep, length):
    mid = row.coord
    start, end = get_start_end(mid, length)
    text = dataset.fetch(row.id, start, end)
    return {key: value[0] for key, value in bert_prep([text]).items()}

def get_target(row: pd.Series):
    return {'target': row.kind}

def get_chroms_1_22(root_directory, pattern):
    paths = expand_dataset_paths(
        '(' + pattern.format('[1..9]') + ',' + pattern.format('[10..22]') + ')', None)
    return [os.path.join(root_directory, path) for path in paths]

class SpliceSiteDataModule(BioNeMoDataModule):

    def __init__(self, cfg, trainer, model):
        self.model = model
        self.train_file = cfg.data.train_file
        self.val_file = cfg.data.val_file
        fasta_directory = cfg.data.fasta_directory
        pattern = cfg.data.fasta_pattern
        chroms = get_chroms_1_22(fasta_directory, pattern)
        # TODO set length from config
        self.length = 400
        # TODO set chroms from config
        self.fasta_dataset = ConcatFastaDataset(
            chroms, self.length, backend='memory',
        )
        super().__init__(cfg, trainer)
        self.init_num_samples()

    def train_dataset(self):
        gff_dataset = self._create_dataset(self.train_file)
        return gff_dataset

    def val_dataset(self):
        gff_dataset = self._create_dataset(self.val_file)
        return gff_dataset

    def test_dataset(self):
        gff_dataset = self._create_dataset(self.val_file)
        return gff_dataset

    def _create_dataset(self, filename):
        bert_prep = KmerBertCollate(
            self.model.tokenizer,
            # TODO can make modify_percent configurable
            modify_percent=0,
            # TODO make 512 configurable
            seq_length=512,
            pad_size_divisible_by_8=True,
        ).collate_fn

        gff_dataset = DataFrameTransformDataset(
            filename,
            functions = [
                partial(
                    fetch_bert_dna,
                    dataset=self.fasta_dataset,
                    bert_prep=bert_prep,
                    # TODO make length configurable
                    length=400
                ),
                get_target,
            ],
            read_csv_args={'dtype': {'id': str}}
        )

        return gff_dataset


    def sample_train_dataset(self, dataset):
        """Creates an upsampled version of the DNABERT training dataset

        """
        num_samples = self.train_num_samples
        # TODO make this configurable from config
        dataset_dir = '/tmp' #os.path.join(self.cfg.dataset_path, 'train',)
        dataset = NeMoUpsampling(
            dataset, num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.dataset.train,
            index_mapping_dir=dataset_dir,
            name='train',
            )

        return dataset

    def sample_val_dataset(self, dataset):
        num_samples = self.val_num_samples
        # TODO make this configurable from config
        dataset_dir = '/tmp' #os.path.join(self.cfg.dataset_path, 'val',)
        dataset = NeMoUpsampling(
            dataset, num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.dataset.val,
            index_mapping_dir=dataset_dir,
            name='val',
            )
        return dataset

