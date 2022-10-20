from typing import Union, Callable, List
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import pad
from pytorch_lightning import LightningDataModule
import pandas as pd

from nemo.collections.common.tokenizers import TokenizerSpec

class SingleValuePredictionDataset(Dataset):
    def __init__(self, 
                 data_file: Union[str, bytes, os.PathLike], 
                 transform_fn: Callable[[pd.Series], torch.Tensor], 
                 input_column: str = 'SMILES', 
                 target_column: str = 'y', 
                 target_dtype: str = 'float32'):
        """
        Dataset that takes a SMILES-like CSV file as an input and tokenizes, encodes, and returns embeddings for each sample.
        
        Params
            data_file: CSV file path
            transform_fn: Callable that encodes a Pandas Series into embeddings
            input_column: String name of feature column in data_file
            target_column: String name of target column in data_file
            target_dtype: String data type for target column
        """
        
        df = pd.read_csv(data_file)
        self.embeddings = transform_fn(df[input_column])
        self.targets = df[target_column].to_numpy().astype(target_dtype)
        self.length = self.targets.shape[0]
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return (self.embeddings[idx], self.targets[idx])     

class SingleValuePredictionDataModule(LightningDataModule):
    def __init__(self, 
                 data_path: Union[str, bytes, os.PathLike],
                 dset_name: str,
                 transform_fn: Callable[[pd.Series], torch.Tensor], 
                 batch_size: int = 32,
                 test_fraction: float = 0.2,
                 input_column: str = 'SMILES', 
                 target_column: str = 'y', 
                 target_dtype: str = 'float32',
                 random_seed: int = 2147483647
                 ):
        """
        LightningDataModule wrapping SingleValuePredictionDataset
        
        Params
            data_path: Path to dataset
            dset_name: String
            transform_fn: Callable that encodes a Pandas Series into embeddings
            batch_size: int
            test_fraction: float
            input_column: String name of feature column
            target_column: String name of target column
            target_dtype: String data type for target column
        """
        
        super().__init__()
        self.data_path = data_path
        self.dset_name = dset_name
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.test_fraction = test_fraction
        self.input_column = input_column
        self.target_column = target_column
        self.target_dtype = target_dtype
        self.random_seed = random_seed

    def setup(self, **kwargs):
        self.full_dataset = SingleValuePredictionDataset(self.data_path,
                                                         self.transform_fn,
                                                         self.input_column, 
                                                         self.target_column,
                                                         self.target_dtype)
        
        test_size = int(self.test_fraction * len(self.full_dataset))
        train_size = len(self.full_dataset) - test_size
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, 
                                                            [train_size, test_size],
                                                            generator=torch.Generator().manual_seed(self.random_seed))
        
    def train_dataloader(self):        
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          generator=torch.Generator().manual_seed(self.random_seed))
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          generator=torch.Generator().manual_seed(self.random_seed))
