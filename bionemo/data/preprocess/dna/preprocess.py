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
import gzip
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bionemo.data.fasta_dataset import ConcatDataset
from bionemo.data.dna.splice_site_dataset import ChrSpliceSitesDataset
from bionemo.utils.remote import GRCh38Ensembl99ResourcePreparer
from bionemo.data.dna.splice_site_dataset import get_autosomes
from bionemo.utils.gff import parse_gff3, build_donor_acceptors_midpoints

def _gunzip(i, o):
    if os.path.exists(i):
        with gzip.open(i, 'rb') as f_in:
            with open(o, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


class SpliceSitePreprocess(object):
    def __init__(self, root_directory, dataset_name):
        """Downloads and preprocesses Ensembl GRCh38 reference genome and
        annotated splice sites

        Args:
            root_directory (str): Directory to store the dataset
            dataset_name (str): A sub-directory to store the train/val/test
                split
        """
        self.root_directory = root_directory
        self.dataset_name = dataset_name
        self.gff_gz_template = 'Homo_sapiens.GRCh38.99.chromosome.{}.gff3.gz'
        self.gff_template = self.gff_gz_template[:-3]
        self.fa_template = 'Homo_sapiens.GRCh38.dna.chromosome.{}.fa.gz'
        self.ensembl_directory = os.path.join(self.root_directory, 'GRCh38.ensembl.99')
        self.chrs = list(range(1, 23))
        self.train_perc = 0.8
        self.val_perc = 0.1
        self.test_perc = 1 - self.train_perc - self.val_perc
        self.size = 30000
        self.sizes = [0, 0, 0]
        for i in range(self.size):
            self.sizes[i % 3] += 1

    def prepare_dataset(self):
        """Downloads and preprocesses reference and splice site dataset
        """
        self.download()
        df = self.make_sites_df()
        self.write_train_val_test_split(df)

    def download(self):
        """Download the GRCh38 Ensembl99 reference and GFF data
        """
        preparer = GRCh38Ensembl99ResourcePreparer(root_directory=self.root_directory)
        preparer.prepare()

        gff_gzs = get_autosomes(self.ensembl_directory, self.gff_gz_template)
        gffs = get_autosomes(self.ensembl_directory, self.gff_template)
        for gff_gz, gff in zip(gff_gzs, gffs):
            _gunzip(gff_gz, gff)

    def make_sites_df(self):
        """Converts the GFF files to a dataframe of donor/acceptor/TN sites

        Returns:
            pd.DataFrame: Contains splice sites
        """
        datasets = []
        for chr in tqdm(self.chrs):
            gff_filename = os.path.join(
                self.ensembl_directory,
                self.gff_template.format(chr),
            )
            gff_contents = parse_gff3(gff_filename)
            donor_acceptor_midpoints = build_donor_acceptors_midpoints(gff_contents)
            datasets.append(ChrSpliceSitesDataset(donor_acceptor_midpoints, str(chr)))
        all_sites = ConcatDataset(datasets)
        df = pd.DataFrame([all_sites[i] for i in range(len(all_sites))])
        return df

    def write_train_val_test_split(self, df):
        """Performs a train/val/test split of the splice site dataframe

        Args:
            df (pd.DataFrame): Dataframe to split
        """
        df0, df1, df2 = df.loc[df.kind == 0], df.loc[df.kind == 1], df.loc[df.kind == 2]
        np.random.seed(724)
        # randomly downsample from each class
        indices0 = np.random.choice(len(df0), size=self.sizes[0], replace=False)
        indices1 = np.random.choice(len(df1), size=self.sizes[1], replace=False)
        indices2 = np.random.choice(len(df2), size=self.sizes[2], replace=False)
        sites_sample_df = pd.concat([df0.iloc[indices0], df1.iloc[indices1], df2.iloc[indices2]])
        # train val test split
        n_total = len(df)
        n_sampled = len(sites_sample_df)
        train_ub = int(self.train_perc * n_sampled)
        val_ub = int(self.val_perc * n_sampled) + train_ub
        test_ub = n_sampled
        np.random.seed(825)
        shuffled_indices = np.arange(n_total)
        np.random.shuffle(shuffled_indices)
        train_df = df.iloc[shuffled_indices[:train_ub]]
        val_df = df.iloc[shuffled_indices[train_ub:val_ub]]
        test_df = df.iloc[shuffled_indices[val_ub:test_ub]]
        # save train val test split
        datadir = Path(self.root_directory) / self.dataset_name
        os.makedirs(datadir, exist_ok=True)
        train_df.to_csv(datadir / 'train.csv')
        val_df.to_csv(datadir / 'val.csv')
        test_df.to_csv(datadir / 'test.csv')
