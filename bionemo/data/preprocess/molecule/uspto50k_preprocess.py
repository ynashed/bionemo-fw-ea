import os
import shutil
from typing import Optional

import pandas as pd
from nemo.utils import logging
from rdkit import Chem

from bionemo.data.utils import download_dataset_from_ngc


class USPTO50KPreprocess:
    """
    Downloads and prepares the reaction data USPTO50K for model training, ie by cleaning and splitting
    the data into train, validation and tests sets.
    """

    def __init__(self, data_dir: str, max_smiles_length: Optional[str] = None):
        self.data_dir = data_dir
        self.max_smiles_length = max_smiles_length
        self.download_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.datapath_raw = None
        self.data_file = 'data.csv'
        self.splits = ['train', 'val', 'test']

    def get_split_dir(self, split: str) -> str:
        return os.path.join(self.processed_dir, split)

    def prepare_dataset(self, ngc_dataset_id: int, filename_raw: str, force: bool = False):
        """
        Downloads reaction dataset and splits it into train, validation, and test sets.
        Args:
            ngc_dataset_id (str): NGC dataset id of the data to be downloaded.
            filename_raw (str): the name of the file with the raw data
            force (bool): if the generation of the dataset should be forced
        """
        if os.path.exists(self.processed_dir) and not force:
            logging.info(f'Path to the processed dataset {self.processed_dir} exists!')
            return

        self.datapath_raw = self.download_raw_data_file(ngc_dataset_id=ngc_dataset_id, filename=filename_raw)

        if self.datapath_raw:
            self.train_val_test_split(datapath_raw=self.datapath_raw)
        else:
            logging.error(f"Failed to download dataset from NGC. Dataset id: {ngc_dataset_id}!")

    def download_raw_data_file(self, ngc_dataset_id: int, filename: Optional[str] = None) -> Optional[str]:
        """
        Downloads raw data from the url link and saves it in a local directory
        Args:
            ngc_dataset_id (int): NGC dataset id of the data to be downloaded.
            filename (str): optional, the name of the file with raw data
        Returns:
            local_filepath (str): path to the location of the downloaded file
        """
        local_filepath = os.path.join(self.download_dir, filename)
        if os.path.exists(local_filepath):
            logging.info(f'File {filename} is already downloaded...')
            return local_filepath

        logging.info(f'Downloading dataset with id {ngc_dataset_id} from NGC to {local_filepath}...')
        os.makedirs(self.download_dir, exist_ok=True)
        try:
            temp_dir = download_dataset_from_ngc(ngc_dataset_id=ngc_dataset_id, dest=self.data_dir)
            temp_filepath = os.path.join(temp_dir, filename)
            shutil.copy(temp_filepath, local_filepath)
            shutil.rmtree(temp_dir)
            logging.info('Download complete.')
            return local_filepath

        except Exception as e:
            logging.error(
                f'Could not download from NGC dataset with id {ngc_dataset_id}: {e}')
            raise e

    def train_val_test_split(self, datapath_raw: str):
        """
        Splits downloaded raw dataset into train, validation and tests sets
        Args:
            datapath_raw (str): local path to the file with the raw data
        """
        logging.info(f'Splitting file {datapath_raw} into {", ".join(self.splits)} data and '
                     f'saving in {self.processed_dir}')

        df = pd.read_pickle(datapath_raw)
        assert all([col in df for col in ['reactants_mol', 'products_mol', 'reaction_type', 'set']])

        # TODO parallelize in the future!
        for name in ['reactants', 'products']:
            df[f'{name}_correct'] = df[f'{name}_mol'].apply(
                lambda mol: True if Chem.MolToSmiles(mol, canonical=True) else False)

        df = df[df['reactants_correct'] & df['products_correct']]
        for name in ['reactants', 'products']:
            df[name] = df[f'{name}_mol'].apply(lambda mol: Chem.MolToSmiles(mol, canonical=False))
            df[f'{name}_len'] = df[f'{name}'].apply(lambda smi: len(smi))

        df.dropna(axis=0, how='any', subset=['reactants', 'products'], inplace=True)
        if self.max_smiles_length:
            df = df[(df['reactants_len'] <= self.max_smiles_length) & (df['products_len'] <= self.max_smiles_length)]
        df.drop(columns=set(df.columns) - {'reactants', 'products', 'set', 'reaction_type'}, inplace=True)
        df.set.replace(to_replace='valid', value='val', inplace=True)

        for split in self.splits:
            df_tmp = df[df['set'] == split]
            df_tmp.reset_index(drop=True, inplace=True)
            dir_tmp = self.get_split_dir(split)
            if not os.path.exists(dir_tmp):
                os.makedirs(dir_tmp, exist_ok=True)
            df_tmp.to_csv(f'{dir_tmp}/{self.data_file}', index=False)
            with open(f'{dir_tmp}/metadata.txt', 'w') as f:
                f.write(f"file size: {df_tmp.shape[0]} \n")
