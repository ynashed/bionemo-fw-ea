# Copyright (c) 2023, NVIDIA CORPORATION.
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
from typing import Optional
import urllib.request
import re


__all__ = ['FLIPSSPreprocess']

ROOT_DIR = '/tmp/flip_ss'

import re
from typing import Dict, List

def get_attributes_from_seq(sequences: List) -> Dict[str, Dict[str, str]]:
    """
    :param sequences: a list of SeqRecords
    :return: A dictionary of ids and their attributes
    """
    result = dict()
    for sequence in sequences:
        result[sequence.name] = {key: value for key, value in re.findall(r"([A-Z_]+)=(-?[A-z0-9]+-?[A-z0-9]*[.0-9]*)", 
                                                                         sequence.description)}
    return result

class FLIPSSPreprocess(object):
    def __init__(self, 
                 root_directory: Optional[str] = ROOT_DIR,
                 seq_filename="sequences.fasta", 
                 labels_filename="sampled.fasta",
                 resolved_filename="resolved.fasta"
                 ):
        super().__init__()
        self.root_directory = pathlib.Path(root_directory) 
        self.seq_filename = seq_filename
        self.labels_filename = labels_filename
        self.resolved_filename = resolved_filename  

    def download_FLIP_data(self, download_dir, url):
        filenames = [self.seq_filename, self.labels_filename, self.resolved_filename]
        exists = os.path.exists(download_dir)
        if not exists:
            os.makedirs(download_dir)
        all_file_paths = []
        for filename in filenames:
            file_url = url + filename
            file_path = os.path.join(download_dir, filename)
            all_file_paths.append(file_path)
            req = urllib.request.Request(
                file_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
                },
            )
            response = urllib.request.urlopen(req)
            with open(file_path, "w") as f:
                f.write(response.read().decode("utf-8"))
            logging.info(f"{filename} downloaded successfully!")
        
        return all_file_paths

    def get_train_val_test_splits(self, splits):
        splits_dict = get_attributes_from_seq(splits)

        train_samples = []
        test_samples = []
        val_samples = []
        for key, label in splits_dict.items():
            if label["SET"] == "train" and label["VALIDATION"] == "False":
                train_samples.append(key)
            elif label["SET"] == "train" and label["VALIDATION"] == "True":
                val_samples.append(key)
            elif label["SET"] == "test":
                test_samples.append(key)

        return train_samples, val_samples, test_samples
    
    def write_csv_files(self, 
                        train_samples, 
                        val_samples, 
                        test_samples, 
                        sequence_indexer,
                        labels_indexer,
                        resolved_indexer,
                        num_csv_files,
                        output_dir):
        name_to_seqs = {seq.name: str(seq.seq) for seq in sequence_indexer}
        name_to_masks = {mask.name: str(mask.seq) for mask in resolved_indexer}
        name_to_labels = {label.name: str(label.seq) for label in labels_indexer}
        for split_name, record_id_list in zip(['train', 'val', 'test'], [train_samples, val_samples, test_samples]):
            logging.info(f'Saving {split_name} split...')

            for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files)):
                logging.debug(f'Writing file number {file_index}...')
                self._csv_files_writer(record_id_list=record_id_split, 
                                      file_index=file_index, 
                                      split_name=split_name, 
                                      seq_dict=name_to_seqs,
                                      labels_dict=name_to_labels,
                                      masks_dict=name_to_masks, 
                                      output_dir=output_dir)
        return

    def _csv_files_writer(self, 
                          record_id_list, 
                          file_index,
                          split_name, 
                          seq_dict,
                          labels_dict, 
                          masks_dict,
                          output_dir, 
                          delimiter=","):
        split_path = os.path.join(output_dir, split_name)
        pathlib.Path(split_path).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(split_path, f'x{str(file_index).zfill(3)}.csv')

        with open(file_name, 'w') as fh:
            header_str = delimiter.join(['id', 'sequence', '3state', 'resolved'])
            fh.write(header_str + '\n')
            for record_id in record_id_list:
                sequence = seq_dict[record_id]
                labels = labels_dict[record_id]
                mask = masks_dict[record_id]
                output = delimiter.join([record_id, sequence, labels, str(mask)]) 
                fh.write(output + '\n')
        return       

    def prepare_dataset(self,
                        output_dir,
                        url="http://data.bioembeddings.com/public/FLIP/fasta/secondary_structure/",
                        num_csv_files=1,
                        ):
        """Download FLIP secondary structure dataset and split into train, valid, and test sets.
        Splits are pre-defined.

        Args:
            url (str): URL for FLIP SS dataset location.
            num_csv_files (int): Number of CSV files to create for each train/val/test split.
        """
        download_dir = self.root_directory.joinpath('raw')
        exists = os.path.exists(output_dir)
        if not exists:
            os.makedirs(output_dir)
        seq_path, labels_path, resolved_path = self.download_FLIP_data(download_dir=download_dir, url=url)
        sequence_indexer = pyfastx.Fasta(seq_path, build_index=True, uppercase=True)
        labels_indexer = pyfastx.Fasta(labels_path, build_index=True, uppercase=True)
        resolved_indexer = pyfastx.Fasta(resolved_path, build_index=True, uppercase=True)
        logging.info('FLIP data download complete.')

        logging.info('Processing FLIP dataset.')
        train_samples, val_samples, test_samples = self.get_train_val_test_splits(labels_indexer)
        
        logging.info(f'Writing processed dataset files to {output_dir}...')
        self.write_csv_files(train_samples=train_samples, 
                             val_samples=val_samples, 
                             test_samples=test_samples, 
                             num_csv_files=num_csv_files, 
                             sequence_indexer=sequence_indexer,
                             labels_indexer=labels_indexer,
                             resolved_indexer=resolved_indexer, 
                             output_dir=output_dir)
        logging.info(f'FLIP dataset preprocessing completed')
