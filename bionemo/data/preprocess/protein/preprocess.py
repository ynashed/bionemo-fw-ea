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

import numpy as np
import pyfastx
import pathlib
from nemo.utils import logging
import os
import requests
import gzip
import shutil
from typing import Optional

from bionemo.data.utils import download_registry_from_ngc, get_ngc_registry_file_list, verify_checksum_matches

__all__ = ['UniRef50Preprocess']

ROOT_DIR = '/tmp/uniref50'
MD5_CHECKSUM = 'e619d3689749562d743f8ecf29a7a7c2'

class UniRef50Preprocess(object):
    def __init__(self, root_directory: Optional[str] = ROOT_DIR, checksum: Optional[str] = MD5_CHECKSUM) -> None:
        """Prepocess UniRef50 data for pre-training. 

        Args:
            root_directory (Optional[str]): Directory for download. Defaults to /tmp/uniref50.
            checksum (Optional[str]): Checksum for file


        Data are downloaded to root_directory/raw (/tmp/uniref50/raw). The split data can be found in
        root_directory/processed.
        """
        super().__init__()
        self.root_directory = pathlib.Path(root_directory)
        self.checksum = checksum

    def _process_file(self, url, download_dir):
        """Download UniRef50 file and unzip

        Args:
            url (str): URL for UniRef50 location.
            download_dir (str): Download directory for UniRef50 file.

        Returns:
            str: Path to UniRef50 FASTA file
        """
        assert url.endswith('.fasta.gz'), AssertionError(f'Expected URL to end with `.fasta.gz`, got {url}..')

        filename, gz_ext = os.path.splitext(os.path.split(url)[-1]) # gz extension
        filename, fasta_ext = os.path.splitext(filename) # fasta extension

        file_path = os.path.join(download_dir, filename + fasta_ext)
        tmp_file_path = os.path.join(download_dir, filename + '_tmp' + fasta_ext)

        gz_file_path = file_path + gz_ext
        tmp_gz_file_path = tmp_file_path + gz_ext

        if os.path.exists(file_path):
            logging.info(f'{url} already exists at {file_path}...')
            return file_path

        logging.info(f'Downloading file to {gz_file_path}...')
        try:
            if not os.path.exists(gz_file_path):
                # Download gzipped file from url
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(tmp_gz_file_path, 'wb') as f:
                        for chunk in r.raw.stream(1024, decode_content=False):
                            if chunk:
                                f.write(chunk)

                os.rename(tmp_gz_file_path, gz_file_path)

            # Extract gzipped file and clean up
            logging.info(f'Extracting file to {file_path}...')
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(tmp_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.rename(tmp_file_path, file_path)
            os.remove(gz_file_path)

            return file_path

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.error(f'{url} Not found')
                return
            else:
                logging.error(
                    f'Could not download file {url}: {e.response.status_code}')
                raise e

    def process_files_uniprot(self, url, download_dir):
        """Download the UniRef50 fasta file and decompress it.

        Parameters:
            url (str): URL for UniRef50 location.
            download_dir (str): Download directory for UniRef50 file.
        """

        logging.info(
            f'Data processing can take an hour or more depending on system resources.')

        logging.info(
            f'Downloading file from {url}...')

        os.makedirs(download_dir, exist_ok=True)
        file_path = self._process_file(url=url, download_dir=download_dir)
        return file_path

    @staticmethod
    def process_files_ngc(ngc_registry_target, ngc_registry_version, download_dir, output_dir, checksum):
        # TODO create base preprocessing class with this method
        assert os.environ.get('NGC_CLI_API_KEY', False), AssertionError("""NGC API key not defined as environment variable "NGC_CLI_API_KEY".
                                                                           Aborting resource download.""")
        ngc_org = os.environ.get('NGC_CLI_ORG', None)
        assert ngc_org, AssertionError('NGC org must be defined by the environment variable NGC_CLI_ORG')
        ngc_team  = os.environ.get('NGC_CLI_TEAM', None)

        # Check if resource already exists at final destination
        file_list = get_ngc_registry_file_list(ngc_registry_target, ngc_registry_version, ngc_org, ngc_team)        
        file_exists = False
        if len(file_list) > 1:
            logging.info(f'Checksum verification not supported if resource contains more than one file.')
        else:
            file_name = file_list[0]
            output_path = os.path.join(output_dir, file_name)
            if os.path.exists(output_path):
                file_exists = True if verify_checksum_matches(output_path, checksum) else False
        
        # Download resource and copy if needed
        if not file_exists:
            os.makedirs(download_dir, exist_ok=True)
            tmp_download_path = download_registry_from_ngc(ngc_registry_target=ngc_registry_target, 
                                                           ngc_registry_version=ngc_registry_version, 
                                                           ngc_org=ngc_org, 
                                                           ngc_team=ngc_team,
                                                           dest=download_dir,
                                                           expected_checksum=checksum)

            # Move to destination directory and clean up
            file_name = os.path.basename(tmp_download_path)
            output_path = os.path.join(output_dir, file_name) # Ensures output_path is defined when file is downloaded
            shutil.copyfile(tmp_download_path, output_path)
            logging.info(f'Download complete at {output_path}.')
        else:
            logging.info(f'File download skipped because file exists at {output_path} and has expected checksum.')

        return output_path   

    @staticmethod
    def _index_fasta_data(fasta_indexer, val_size, test_size, random_seed):
        """Create index lists for train, validation, and test splits

        Args:
            fasta_indexer (pyfastx): Memory mapped index of UniRef50 FASTA file
            val_size (int): Number of protein sequences to put in validation set.
            test_size (int): Numter of protein sequences to put in test set.
            random_seed (int): Random seed.

        Returns:
            List of indexes: list of train, validation, test indexes
        """
        sample_list = np.arange(len(fasta_indexer))
        
        rng = np.random.default_rng(random_seed)
        rng.shuffle(sample_list)

        val_samples = sample_list[:val_size]
        test_samples = sample_list[val_size:val_size + test_size]
        train_samples = sample_list[val_size + test_size:]

        assert len(val_samples) == val_size, AssertionError('Validation dataset is not the correct size.')
        assert len(test_samples) == test_size, AssertionError('Test dataset is not the correct size.')
        assert len(fasta_indexer) - len(val_samples) - len(test_samples) == len(train_samples), AssertionError('Train dataset is not the correct size.')

        return train_samples, val_samples, test_samples

    @staticmethod
    def _protein_sequence_filewriter(fasta_indexer, record_id_list, file_index, split_name, output_dir, delimiter=','):
        """CSV file writer for FASTA data

        Args:
            fasta_indexer (pyfastx): Memory mapped index of UniRef50 FASTA file
            record_id_list (Numpy array): array of file indexes for the splits
            file_index (int): Index number of the filename.
            split_name (str): Directory name for the split -- "train", "val", "test"
            output_dir (str): Output directory for CSV data.
            delimiter (str, optional): CSV delimiter. Defaults to ','.
        """
        
        split_path = os.path.join(output_dir, split_name)
        pathlib.Path(split_path).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(split_path, f'x{str(file_index).zfill(3)}.csv')

        with open(file_name, 'w') as fh:
            header_str = delimiter.join(['record_id', 'record_name', 'sequence_length', 'sequence'])
            fh.write(header_str + '\n')
            for record_id in record_id_list:
                record = fasta_indexer[record_id]
                output = delimiter.join([str(record.id), record.name, str(len(record.seq)), record.seq])
                fh.write(output + '\n')
        return

    def train_val_test_split(self, train_samples, val_samples, test_samples, num_csv_files, fasta_indexer, output_dir):
        """Create CSV files for train, val, test data

        Args:
            train_samples (numpy array): Array of index numbers for training data.
            val_samples (numpy array): Array of index numbers for validation data
            test_samples (numpy array): Array of index numbers for test data
            num_csv_files (int): Number of CSV files to create for each train/val/test split.
            fasta_indexer (pyfastx): Memory mapped index of UniRef50 FASTA file
            output_dir (str): Output directory for CSV data.
        """
        
        for split_name, record_id_list in zip(['train', 'val', 'test'], [train_samples, val_samples, test_samples]):
            logging.info(f'Creating {split_name} split...')

            for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files)):
                logging.debug(f'Writing file number {file_index}...')
                self._protein_sequence_filewriter(record_id_list=record_id_split, 
                                                  file_index=file_index, 
                                                  split_name=split_name, 
                                                  fasta_indexer=fasta_indexer, 
                                                  output_dir=output_dir)
        return

    def prepare_dataset(self,
                        ngc_registry_target=None,
                        ngc_registry_version=None,
                        url='https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz',
                        source='ngc',
                        output_dir=None,
                        num_csv_files=50,
                        val_size=5000,
                        test_size=1000000,
                        random_seed=0):
        """Download UniRef50 dataset and split into train, valid, and test sets.

        Args:
            url (str): URL for UniRef50 location.
            num_csv_files (int): Number of CSV files to create for each train/val/test split.
            val_size (int): Number of protein sequences to put in validation set.
            test_size (int): Number of protein sequences to put in test set.
            random_seed (int): Random seed.
        """

        logging.info('Download and preprocess of UniRef50 data does not currently use GPU. Workstation or CPU-only instance recommended.')

        download_dir = self.root_directory.joinpath('raw')
        if output_dir is None:
            output_dir = self.root_directory.joinpath('processed')
        os.makedirs(output_dir, exist_ok=True)

        if source == 'ngc':
            assert ngc_registry_target is not None
            assert ngc_registry_version is not None
            file_path = self.process_files_ngc(ngc_registry_target=ngc_registry_target, 
                                               ngc_registry_version=ngc_registry_version, 
                                               download_dir=download_dir,
                                               output_dir=output_dir,
                                               checksum=self.checksum)

        elif source == 'uniprot':
            file_path = self.process_files_uniprot(url=url, download_dir=download_dir)

        logging.info('UniRef50 data processing complete.')

        logging.info('Indexing UniRef50 dataset.')
        fasta_indexer = pyfastx.Fasta(file_path, build_index=True, uppercase=True)  
        train_samples, val_samples, test_samples = self._index_fasta_data(fasta_indexer=fasta_indexer,
                                                                          val_size=val_size,
                                                                          test_size=test_size,
                                                                          random_seed=random_seed)

        logging.info(f'Writing processed dataset files to {output_dir}...')
        self.train_val_test_split(train_samples=train_samples, 
                                  val_samples=val_samples, 
                                  test_samples=test_samples, 
                                  num_csv_files=num_csv_files, 
                                  fasta_indexer=fasta_indexer, 
                                  output_dir=output_dir)
