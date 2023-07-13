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

import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from bionemo.core import BioNeMoDataModule
from bionemo.tokenizer.label2id_tokenizer import Label2IDTokenizer
from bionemo.data.datasets.per_token_value_dataset import get_data
from bionemo.model.utils import _reconfigure_inference_batch
from bionemo.data.utils import expand_dataset_paths
from tqdm import trange
import os


class SingleValueDataset(Dataset):
    def __init__(self, datafiles, max_seq_length, emb_batch_size=None, 
                 model=None, input_column: str='SMILES', 
                 target_column: str='y', task: str="regression"):
        dfs = []
        for file in datafiles:
            dfs.append(pd.read_csv(file))
        df = pd.concat(dfs)
        self.max_seq_length = max_seq_length
        self.input_column = input_column
        self.target_column = target_column
        self.model = model
        self.emb_batch_size = emb_batch_size
        self.input_seq = df[self.input_column].tolist()
        self.labels = df[self.target_column].tolist()
        seq_lengths = np.array([len(s) for s in self.input_seq])
        self.input_seq = np.array(self.input_seq)[seq_lengths <= (self.max_seq_length - 2)]
        self.labels = np.array(self.labels)[seq_lengths <= (self.max_seq_length - 2)]
        self.embeddings = []
        self.masks = []
        if model is not None:
            self.encoder_model_batch_size = model.cfg.model.micro_batch_size
            with torch.no_grad():
                _reconfigure_inference_batch(self.emb_batch_size) 

                batches = list(range(0, len(self.labels), self.emb_batch_size))
                if batches[-1] < len(self.labels):
                    batches.append(len(self.labels))
                for i in trange(len(batches) - 1):
                    embeddings = self.compute_embeddings(self.input_seq[batches[i]:batches[i+1]])
                    self.embeddings += list(embeddings.cpu().float())
                _reconfigure_inference_batch(self.encoder_model_batch_size, global_batch_size=self.model.cfg.model.global_batch_size)

    def compute_embeddings(self, seqs):
        if self.model is not None:
            embeddings = self.model.seq_to_embeddings(seqs)
            return embeddings 
        else:
            return seqs

    def get_embeddings(self, idx):
        if self.model is not None:
            return self.embeddings[idx]
        else:
            return self.input_seq[idx]

    def get_labels(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
    
        embeddings = self.get_embeddings(idx)
        labels = self.get_labels(idx)
        item_dict = {
            "embeddings": embeddings,
            "target": labels
            }
        return item_dict
    
    def free_memory(self):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_")]
        for m in members:
            self.__delattr__(m)
        gc.collect()
    
    def prepare_batch(self, batch, data):
        return batch['embeddings'].float().to("cuda"), batch['target'].float().to("cuda")


class SingleValueDataModule(BioNeMoDataModule):
    def __init__(self, cfg, trainer, model):
        super().__init__(cfg, trainer)
        self.model = model
        self.parent_cfg = cfg
        if self.cfg.task_type not in ["regression", "classification"]:
            raise ValueError("Invalid task_type was provided {}. " + \
                             "Supported task_type: 'classification' and 'regression'".format(self.cfg.task))
        if self.cfg.task_type == "classification":
            self.tokenizer = Label2IDTokenizer()
        else:
            self.tokenizer = None

    def _update_tokenizer(self, tokenizer, labels):
        tokenizer = tokenizer.build_vocab(labels)
        return tokenizer

    def _create_dataset(self, split, files):
        datafiles = os.path.join(self.cfg.dataset_path,
                                 self.cfg.task_name, 
                                 split, 
                                 files)
        datafiles = expand_dataset_paths(datafiles, ".csv")
        dataset = SingleValueDataset(
            datafiles=datafiles, 
            max_seq_length=self.parent_cfg.seq_length,
            emb_batch_size=self.cfg.emb_batch_size,
            model=self.model, 
            input_column=self.cfg.sequence_column, 
            target_column=self.cfg.target_column
            )
        if self.tokenizer is not None:
            self.tokenizer = self._update_tokenizer(
                self.tokenizer, 
                dataset.labels.reshape(-1, 1)
                )
            dataset.labels = get_data._tokenize_labels([self.tokenizer], dataset.labels.reshape(1, 1, -1), [self.cfg.num_classes])[0][0] 
        return dataset

    def train_dataset(self):
        """Creates a training dataset
        Returns:
            Dataset: dataset to use for training
        """
        self.train_ds = self._create_dataset("train", 
                                             self.cfg.dataset.train)
        return self.train_ds

    def val_dataset(self):
        """Creates a validation dataset
        Returns:
            Dataset: dataset to use for validation
        """
        if "val" in self.cfg.dataset:
            self.val_ds = self._create_dataset("val", 
                                             self.cfg.dataset.val)
            return self.val_ds
        else:
            pass

    def test_dataset(self):
        """Creates a testing dataset
        Returns:
            Dataset: dataset to use for testing
        """
        if "test" in self.cfg.dataset:
            self.test_ds = self._create_dataset("test", 
                                                self.cfg.dataset.test)
            return self.test_ds
        else:
            pass
