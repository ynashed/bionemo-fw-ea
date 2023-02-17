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

import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from bionemo.core import BioNeMoDataModule
from bionemo.model.utils import _reconfigure_inference_batch
from tqdm import trange


class FineTuneDataset(Dataset):
    def __init__(self, data_file, emb_batch_size=None, model=None, input_column: str = 'SMILES', target_column: str = 'y'):
        self.data_file = data_file
        df = pd.read_csv(data_file)
        self.input_column = input_column
        self.target_column = target_column
        self.model = model
        self.emb_batch_size = emb_batch_size
        self.input_seq = df[self.input_column].tolist()
        self.labels = df[self.target_column].tolist()
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


class FineTuneDataModule(BioNeMoDataModule):
    def __init__(self, cfg, trainer, model):
        super().__init__(cfg, trainer)
        self.model = model

        self.train_data_path = Path(cfg.data.train_ds.data_file)
        self.val_data_path = Path(cfg.data.validation_ds.data_file)
        if "test_ds" in cfg.data:
            self.test_data_path = Path(cfg.data.test_ds.data_file)
        else:
            self.test_data_path = None

    def train_dataset(self):
        """Creates a training dataset
        Returns:
            Dataset: dataset to use for training
        """
        return FineTuneDataset(
            data_file=self.train_data_path, 
            emb_batch_size=self.cfg.emb_batch_size,
            model=self.model, 
            input_column=self.cfg.smis_column, 
            target_column=self.cfg.target_column
            )

    def val_dataset(self):
        """Creates a validation dataset
        Returns:
            Dataset: dataset to use for validation
        """
        return FineTuneDataset(
            data_file=self.val_data_path, 
            emb_batch_size=self.cfg.emb_batch_size,
            model=self.model, 
            input_column=self.cfg.smis_column, 
            target_column=self.cfg.target_column
            )

    def test_dataset(self):
        """Creates a testing dataset
        Returns:
            Dataset: dataset to use for testing
        """
        if self.test_data_path is not None:
            return FineTuneDataset(
                data_file=self.test_data_path, 
                emb_batch_size=self.cfg.emb_batch_size,
                model=self.model, 
                input_column=self.cfg.smis_column, 
                target_column=self.cfg.target_column
                )
        else:
            pass
