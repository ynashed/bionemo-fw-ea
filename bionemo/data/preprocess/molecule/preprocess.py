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
import sys
import requests
import multiprocessing as mp
import pandas as pd

from datetime import datetime
from subprocess import run
from multiprocessing import Pool
from functools import partial

from rdkit import Chem

from nemo.utils import logging

MAX_LENGTH = 150


__all__ = ['SmilesPreprocess']

class SmilesPreprocess(object):

    def __init__(self) -> None:
        super().__init__()
        self.retry = False

    def _run_cmd(self, cmd, failure_error='Unexpected error while executing bash cmd'):
        logging.debug(f'Running cmd: {cmd}')

        process = run(['bash', '-c', cmd], capture_output=True, text=True)

        if process.returncode != 0:
            logging.error(failure_error)
            sys.exit(process.returncode)
        return process

    def _process_file(self, url, max_smiles_length=512, download_dir='/tmp/zinc15/raw'):

        filename = url.split('/')[-1]
        if os.path.exists(os.path.join(download_dir, filename)):
            logging.info(f'{url} already downloaded...')
            return

        logging.info(f'Downloading file {filename}...')
        num_molecules_filtered = 0
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                tmp_filename = os.path.join(download_dir, filename + '_tmp')
                header = True
                with open(tmp_filename, 'w') as f:
                    for line in r.iter_lines():
                        if header:
                            header = False
                            f.write("zinc_id,SMILES\n")
                            continue
                        line = line.decode("utf-8")
                        splits = line.split("\t")
                        if len(splits) < 2:
                            continue

                        smi, zinc_id = splits[0], splits[1]
                        try:
                            mol = Chem.MolFromSmiles(smi)
                            smi = Chem.MolToSmiles(mol, canonical=True)
                        except RuntimeError:
                            continue

                        if len(smi) > MAX_LENGTH:
                            continue
                        
                        if len(smi) <= max_smiles_length:
                            f.write(f"{zinc_id},{smi}\n")
                        else:
                            num_molecules_filtered += 1

            os.rename(tmp_filename, os.path.join(download_dir, filename))
            if num_molecules_filtered > 0:
                logging.info(f'Filtered {num_molecules_filtered} molecules from {filename} with length longer than {max_smiles_length}')
            return
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.error(f'{url} Not found')
                return
            else:
                logging.error(
                    f'Could not download file {url}: {e.response.status_code}')
                raise e

    def __processing_failure(self, e):
        logging.info(f'Processing failure: {e}')
        self.retry = True

    def process_files(self, links_file, pool_size=8, max_smiles_length=512, download_dir='/tmp/zinc15/raw'):
        """
        Download all the files in the links file.

        Parameters:
            links_file (str): File containing links to be downloaded.
            pool_size (int): Number of processes to use.
            download_dir (str): Directory to download the files to.
        """

        logging.info(
            f'Downloading files from {links_file} with poolsize {pool_size}...')

        os.makedirs(download_dir, exist_ok=True)
        with open(links_file, 'r') as f:
            links = list(set([x.strip() for x in f]))

        download_funct = partial(self._process_file, max_smiles_length=max_smiles_length, download_dir=download_dir)

        while True:
            pool = Pool(processes=pool_size)
            pool.map_async(download_funct,
                           links,
                           error_callback=self.__processing_failure)
            pool.close()
            pool.join()

            if self.retry:
                logging.info(
                    'Retrying to download files that failed with 503...')
                self.retry = False
            else:
                break

    def _process_split(self, datafile, val_frac, test_frac, output_dir='/tmp/zinc15/processed/', seed=0):
        filename = f'{output_dir}/split_data/{datafile}'
        logging.info(f'Splitting file {filename} into train, validation, and test data')

        df = pd.read_csv(filename, header=None, names=['zinc_id', 'smiles'])
        
        # Calculate sample sizes before size of dataframe changes
        test_samples = max(int(test_frac * df.shape[0]), 1)
        val_samples = max(int(val_frac * df.shape[0]), 1)

        test_df = df.sample(n=test_samples, random_state=seed)
        df = df.drop(test_df.index)  # remove test data from training data

        val_df = df.sample(n=val_samples, random_state=seed)
        df = df.drop(val_df.index)  # remove validation data from training data

        df.to_csv(f'{output_dir}/train/{datafile}.csv', index=False)
        test_df.to_csv(f'{output_dir}/test/{datafile}.csv', index=False)
        val_df.to_csv(f'{output_dir}/val/{datafile}.csv', index=False)

        # TODO deprecate this
        with open(f'{output_dir}/val/metadata.csv', 'a') as f:
            f.write(f"{datafile},{val_df.shape[0]}\n")
        with open(f'{output_dir}/test/metadata.csv', 'a') as f:
            f.write(f"{datafile},{test_df.shape[0]}\n")
        with open(f'{output_dir}/train/metadata.csv', 'a') as f:
            f.write(f"{datafile},{df.shape[0]}\n")

        del df
        del test_df
        del val_df

    def train_val_test_split(self, 
                             download_dir, 
                             output_dir, 
                             train_samples_per_file, 
                             val_samples_per_file, 
                             test_samples_per_file, 
                             pool_size=8, 
                             seed=0):
        if os.path.exists(output_dir):
            logging.info(f'{output_dir} already exists...')
            os.rename(output_dir, output_dir +
                      datetime.now().strftime('%Y%m%d%H%M%S'))

        split_data = os.path.join(output_dir, 'split_data')
        os.makedirs(split_data, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

        total_samples_per_file = sum([train_samples_per_file, val_samples_per_file, test_samples_per_file])
        self._run_cmd(f"cd {split_data}; tail -q -n +2 {download_dir}/** | split -d -l {total_samples_per_file} -a 3",
                      failure_error='Error while merging files')

        split_files = os.listdir(split_data)
        logging.info(f'The data has been be split into {len(split_files)} files.')

        with open(f'{output_dir}/val/metadata.csv', 'w') as f:
            f.write(f"file,size\n")
        with open(f'{output_dir}/test/metadata.csv', 'w') as f:
            f.write(f"file,size\n")
        with open(f'{output_dir}/train/metadata.csv', 'w') as f:
            f.write(f"file,size\n")

        val_frac = val_samples_per_file / total_samples_per_file
        test_frac = test_samples_per_file / total_samples_per_file
        with Pool(processes=pool_size) as pool:
            split_funct = partial(self._process_split, val_frac=val_frac, test_frac=test_frac, output_dir=output_dir, seed=seed)

            pool.map(split_funct,
                     split_files)

    def prepare_dataset(self,
                        max_smiles_length=512,
                        train_samples_per_file=10050000,
                        val_samples_per_file=100,
                        test_samples_per_file=50000,
                        links_file='conf/model/data/ZINC-downloader.txt',
                        download_dir='/tmp/zinc15/raw',
                        output_dir='/tmp/zinc15/processed',
                        seed=0):
        """
        Download ZINC15 tranches and split into train, valid, and test sets.

        Parameters:
            max_smiles_length (int): Maximum SMILES length to be used
            train_samples_per_file (int): number of training samples per file. Controls number of shards created.
            val_samples_per_file (int): number of validation samples per file. Controls number of shards created.
            test_samples_per_file (int): number of test samples per file. Controls number of shards created.
            links_file (str): File containing links to be downloaded.
            download_dir (str): Directory to download the files to.
            output_dir (str): Directory to save the processed data to.
            seed (int): Random seed for data splitting
        """

        if os.path.basename(links_file) == 'ZINC-downloader.txt':
                logging.info(
                             f'NOTE: It appears the all ZINC15 tranches have been selected for processing. '\
                               'Processing all of the ZINC15 tranches can require up to a day, depending on resources. '\
                               'To test on a subset set model.data.links_file to ZINC-downloader-test.txt')

        # If 503 errors or deadlocks are a problem, reduce pool size to 8.
        self.process_files(links_file,
                           pool_size=16,
                           max_smiles_length=max_smiles_length,
                           download_dir=download_dir)
        logging.info('Download complete.')

        samples_per_file = [train_samples_per_file, val_samples_per_file, test_samples_per_file]
        logging.info(f'Now splitting the data into train, val, and test sets with {train_samples_per_file}, {val_samples_per_file}, {test_samples_per_file} samples per file, respectively.')

        self.train_val_test_split(download_dir,
                                  output_dir,
                                  train_samples_per_file=train_samples_per_file, 
                                  val_samples_per_file=val_samples_per_file, 
                                  test_samples_per_file=test_samples_per_file, 
                                  pool_size=8,
                                  seed=seed)
