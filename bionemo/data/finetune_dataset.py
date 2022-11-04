import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from nemo.utils import logging
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Union
from torch.nn.utils.rnn import pad_sequence
from bionemo.core import BioNeMoDataModule

class FineTuneDataset(Dataset):
    def __init__(self, data_file: Union[str, bytes, os.PathLike], tokenizer_fn, input_column: str = 'SMILES', target_column: str = 'y'):
        self.data_file = data_file
        self.df = pd.read_csv(data_file)
        self.input_column = input_column
        self.target_column = target_column

        self.tokenizer = tokenizer_fn

        self.token_ids = [self.tokenizer.text_to_ids(t) for t in self.df[self.input_column]]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
    
        target = self.df[self.target_column].iloc[idx]

        #idea: return dictionary which tells the name of target
        #token_ids = self.transform_fn.text_to_ids(smile)

        return {"token_ids": self.token_ids[idx], "target": target}

    #collate function to handle padding of each batch
    def custom_collate(self, data):
        inputs = [torch.tensor(d['token_ids']) for d in data]
        labels = [d['target'] for d in data]
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_id)
        labels = torch.tensor(labels)
        return {
            'token_ids': inputs, 
            'target': labels
        }

class FineTuneDataModule(BioNeMoDataModule):
    def __init__(self, cfg, tokenizer_fn):

        self.data_path = Path(cfg.downstream_task.dataset)
        self.data = FineTuneDataset(self.data_path, tokenizer_fn)

        train_size = int(0.75*len(self.data))
        val_size = len(self.data) - train_size

        self.train_ds, self.val_ds = torch.utils.data.random_split(self.data, [train_size, val_size])

    def train_dataset(self):
        """Creates a training dataset
        Returns:
            Dataset: dataset to use for training
        """
        return self.train_ds

    def val_dataset(self):
        """Creates a validation dataset
        Returns:
            Dataset: dataset to use for validation
        """
        return self.val_ds

    def test_dataset(self):
        """Creates a testing dataset
        Returns:
            Dataset: dataset to use for testing
        """
        raise NotImplementedError()

    def adjust_train_dataloader(self,model,dataloader):
        """Allows adjustments to the training dataloader
        This is a good place to adjust the collate function of the dataloader.
        """

        dataloader.collate_fn = self.data.custom_collate

    def adjust_val_dataloader(self, model, dataloader):
        """Allows adjustments to the validation dataloader
        This is a good place to adjust the collate function of the dataloader.
        """

        dataloader.collate_fn = self.data.custom_collate