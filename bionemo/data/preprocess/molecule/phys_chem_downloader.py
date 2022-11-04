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
import urllib.request
from pathlib import Path

from nemo.utils import logging


__all__ = ['PhysChemPreprocess']

class PhysChemPreprocess(object):
    '''
    Handles download of PhysChem datasets
    '''
    def __init__(self):
        super().__init__()
    
    def prepare_dataset(self,
                        links_file: str,
                        output_dir: str):
        '''
        Downloads Physical Chemistry datasets
        
        Params:
            links_file: File containing links to be downloaded.
            output_dir: Directory to save the processed data to.
        '''
        logging.info(f'Downloading files from {links_file}...')
        
        os.makedirs(output_dir, exist_ok=True)
        with open(links_file, 'r') as f:
            links = list(set([x.strip() for x in f]))
            
        download_dir_path = Path(output_dir)
            
        for url in links:
            filename = url.split('/')[-1]
            if os.path.exists(os.path.join(output_dir, filename)):
                logging.info(f'{url} already downloaded...')
                continue
            
            logging.info(f'Downloading file {filename}...')
            urllib.request.urlretrieve(url, download_dir_path / filename)
        
        logging.info('Download complete.')
