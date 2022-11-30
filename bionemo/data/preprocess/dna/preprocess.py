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
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from omegaconf.omegaconf import OmegaConf
from nemo.utils import logging
from bionemo.data.fasta_dataset import ConcatDataset
from bionemo.data.dna.splice_site_dataset import (
    ChrSpliceSitesDataset,
    get_autosomes,
)
from bionemo.data.utils import gunzip
from bionemo.utils.remote import (
    GRCh38Ensembl99ResourcePreparer,
    GRCh38p13_ResourcePreparer

)
from bionemo.utils.gff import parse_gff3, build_donor_acceptors_midpoints
from bionemo.tokenizer.dna_tokenizer import KmerTokenizer
from bionemo.utils.preprocessors import FastaSplitNsPreprocessor
from bionemo.data.utils import expand_dataset_paths


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
            gunzip(gff_gz, gff, exist_ok=True)

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


# This abstract dataclass helps solve the problem where we want to use
# dataclass to define the constructor for DNABERTPreprocess but also want to
# extend the constructor.
@dataclass
class DNABERTPreprocessorDataClass(object):
    genome_dir: str
    tokenizer_model_path: str
    tokenizer_vocab_path: str
    tokenizer_k: int
    dataset_conf: OmegaConf


class DNABERTPreprocess(DNABERTPreprocessorDataClass):

    def __init__(self, *args, **kwargs):
        """Downloads and preprocesses GRCh38p13 reference genome and constructs
        the tokenizer for DNABERT

        genome_dir (str): Directory to store the reference genome in
        tokenizer_model_path (str): Filepath to store the tokenzier parameters
        tokenizer_vocab_path (str): Filepath to store the tokenizer vocab
        tokenizer_k (int): k-mer size for the tokenizer
        dataset_conf (OmegaConf): has 'train', 'val', 'test' keys containing
            the names of preprocessed train/val/test files to use for training.
        """
        super().__init__(*args, **kwargs)
        self._validate_tokenizer_args(
            self.tokenizer_model_path,
            self.tokenizer_vocab_path,
        )

    def build_tokenizer(self, model_output_name, vocab_output_name, k):
        """Builds a tokenizer for a given k

        Args:
            model_output_name (str): Filepath to store the tokenizer parameters
            vocab_output_name (str): Filepath to store the tokenizer vocab
            k (int): k-mer size for the tokenizer
        """
        tokenizer = KmerTokenizer(k=k)
        tokenizer.build_vocab_from_k()
        tokenizer.save_vocab(
            model_file=model_output_name,
            vocab_file=vocab_output_name,
        )
        return tokenizer

    def _validate_tokenizer_args(self, model_output_name, vocab_output_name):
        model_exists = os.path.exists(model_output_name)
        vocab_exists = os.path.exists(vocab_output_name)
        if model_exists and vocab_exists:
            logging.warning(
                    f'Tokenizer model file: {model_output_name} and tokenizer '
                    f'vocab name: {vocab_output_name} already exist. Skipping '
                    f'tokenizer building stage.'
                )
            return
        elif model_exists:
            raise ValueError(
                    f'Tokenizer model file: {model_output_name} already '
                    f'exists, but vocab file: {vocab_output_name} does not.'
                )
        elif vocab_exists:
            raise ValueError(
                    f'Tokenizer vocab file: {vocab_output_name} already '
                    f'exists, but model file: {model_output_name} does not.'
                )

    def split_train_val_test_chrs(self, preprocessed_files):
        """Splits the preprocessed files into train/val/test dirs

        Args:
            preprocessed_files (List[str]): List of preprocessed files

        Raises:
            ValueError: If the a requested train/val/test file does not exist
                in the preprocessed files
        """
        splits = ['train', 'val', 'test']
        preprocessed_filenames = [
                os.path.basename(f) for f in preprocessed_files
            ]
        for split in splits:
            split_dir = os.path.join(self.genome_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            pattern = self.dataset_conf[split]
            files_from_config = expand_dataset_paths(pattern, '')
            for fa_file in files_from_config:
                # if a preprocessed file matches the pattern in the config
                # for train/val/test, copy it from the genome directory to the
                # appropriate folder for the split, e.g., train
                # Files will not be overwritten if they already exist.
                # this design is so that we can split training by chromosome
                # and can update the training chromosomes along the way.

                try:
                    preprocessed_index = preprocessed_filenames.index(fa_file)
                except ValueError:
                    raise ValueError(
                        f'File: {fa_file} from {split} config: {pattern} not '
                        f'found in {preprocessed_filenames}.')

                file_exists_in_split_dir = os.path.exists(
                    os.path.join(split_dir, fa_file))
                if file_exists_in_split_dir:
                    logging.warning(
                        f'File: {fa_file} not copied to {split} split'
                        f' directory because it already exists.'
                    )
                else:
                    logging.info(
                        f'Copying file: {fa_file} to {split_dir}'
                    )
                    shutil.copy(
                        preprocessed_files[preprocessed_index], split_dir
                    )

    def preprocess(self):
        """Preprocesses for the DNABERT model
        """
        filenames = GRCh38p13_ResourcePreparer(
            dest_dir=self.genome_dir, root_directory='/').prepare()

        preprocessed_files = self.preprocess_fastas(filenames)

        logging.info('Creating train/val/test split.')
        self.split_train_val_test_chrs(preprocessed_files)

        self.build_tokenizer(
            self.tokenizer_model_path, self.tokenizer_vocab_path,
            self.tokenizer_k,
        )

    def preprocess_fastas(self, filenames):
        """Splits fasta files into contigs on N's, which is need for training

        Args:
            filenames (List[str]): List of files to preprocess

        Returns:
            List[str]: List of preprocessed files
        """
        logging.info('Splitting fasta files...')
        fasta_preprocessor = FastaSplitNsPreprocessor(filenames)
        preprocessed_files = []
        for fasta in fasta_preprocessor.get_elements():
            preprocessed_file = fasta_preprocessor.get_chunked_name(fasta)
            if os.path.exists(preprocessed_file):
                logging.warning(f'Splitting skipped: already processed '
                                f'{preprocessed_file}')
            else:
                fasta_preprocessor.apply(fasta)
            preprocessed_files.append(preprocessed_file)
        return preprocessed_files
