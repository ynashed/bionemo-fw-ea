from enum import IntEnum
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict, Tuple
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


class SpliceSite(IntEnum):
    DONOR = 0
    ACCEPTOR = 1
    NEGATIVE = 2


class InstanceDataset(Dataset):
    def __init__(self, list_: List, kind: SpliceSite):
        self.list_ = list_
        self.kind = kind

    def __getitem__(self, idx):
        return {'coord': self.list_[idx], 'kind': int(self.kind)}

    def __len__(self):
        return len(self.list_)


class SiteDataset(Dataset):
    def __init__(self, site_tuple: Tuple[List[int], List[int], List[int]], site: str):
        self.site = site
        ds_0 = InstanceDataset(site_tuple[0], SpliceSite.DONOR)
        ds_1 = InstanceDataset(site_tuple[1], SpliceSite.ACCEPTOR)
        ds_2 = InstanceDataset(site_tuple[2], SpliceSite.NEGATIVE)
        self._dataset = ConcatDataset([ds_0, ds_1, ds_2])

    def __getitem__(self, idx):
        item = self._dataset[idx]
        item.update({'site': self.site})
        return item

    def __len__(self):
        return len(self._dataset)


class ChrSpliceSitesDataset(Dataset):
    def __init__(self, chr_sites: Dict[str, Tuple[List[int], List[int], List[int]]], id_: str):
        self.chr_sites = chr_sites
        self._dataset = ConcatDataset(SiteDataset(sites, name) for name, sites in chr_sites.items())
        self.id_ = id_

    def __getitem__(self, idx):
        item = self._dataset[idx]
        item.update({'id': self.id_})
        return item

    def __len__(self):
        return len(self._dataset)

def get_start_end(coord, length):
    start = int(coord - math.ceil(length / 2)) + 1
    end = int(coord + math.floor(length / 2)) + 1
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
            modify_percent=0,
            seq_length=self.length,
            pad_size_divisible_by_8=True,
        ).collate_fn

        gff_dataset = DataFrameTransformDataset(
            filename,
            functions = [
                partial(
                    fetch_bert_dna,
                    dataset=self.fasta_dataset,
                    bert_prep=bert_prep,
                    length=self.length,
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
        dataset_dir = os.path.join(self.cfg.dataset_path, 'train',)
        dataset = NeMoUpsampling(
            dataset, num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.train_file,
            index_mapping_dir=dataset_dir,
            name='train',
            )

        return dataset

    def sample_val_dataset(self, dataset):
        num_samples = self.val_num_samples
        # TODO make this configurable from config
        dataset_dir = os.path.join(self.cfg.dataset_path, 'val',)
        dataset = NeMoUpsampling(
            dataset, num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.val_file,
            index_mapping_dir=dataset_dir,
            name='val',
            )
        return dataset
