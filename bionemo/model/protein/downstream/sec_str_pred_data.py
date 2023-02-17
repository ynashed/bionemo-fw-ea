# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
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
import numpy as np
from tqdm import tqdm

from tqdm import tqdm, trange

from bionemo.model.utils import _reconfigure_inference_batch
from torch.utils.data import Dataset
from bionemo.core import BioNeMoDataModule

three_state_label2num = {"C": 0, "H": 1, "E": 2}
eight_state_label2num = {"_": 0, "E": 1, "G": 2, "T": 3, "H": 4, "S": 5, "B": 6, "I": 7}

three_state_num2label = {0: "C", 1: "H", 2: "E"}
eight_state_num2label = {0: "_", 1: "E", 2: "G", 3: "T", 4: "H", 5: "S", 6: "B", 7: "I"}


class SSDataModule(BioNeMoDataModule):
    def __init__(self, cfg, trainer, model):
        super().__init__(cfg, trainer)
        self.model=model
    
    def train_dataset(self):
        """Creates a training dataset
        Returns:
            Dataset: dataset to use for training
        """
        self.train_ds = get_data(
            datafile=self.cfg.train_ds.data_file, 
            model=self.model,
            max_seq_length=self.cfg.max_seq_length, 
            emb_batch_size=self.cfg.emb_batch_size
        )
        return SSDataset(self.train_ds)

    def val_dataset(self):
        self.val_ds = get_data(
            datafile=self.cfg.validation_ds.data_file, 
            model=self.model,
            max_seq_length=self.cfg.max_seq_length,
            emb_batch_size=self.cfg.emb_batch_size
        )
        return SSDataset(self.val_ds)

    def test_dataset(self):
        if "test_ds" in self.cfg:
            self.test_ds = get_data(
                datafile=self.cfg.test_ds.data_file, 
                model=self.model,
                max_seq_length=self.cfg.max_seq_length,
                emb_batch_size=self.cfg.emb_batch_size
            )
            return SSDataset(self.test_ds)
        else:
            pass


class SSDataset(Dataset):
    """Read data and convert into batches, labels"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.length()

    def __getitem__(self, idx):
        embeddings = self.data.get_embeddings(idx)
        # TODO: check if should be max_length - 2
        if isinstance(embeddings, str):
            seq_len = len(embeddings)
        else:
            seq_len, emb_dim = embeddings.size()
            embeddings = torch.cat([embeddings, torch.zeros((self.data.max_length - seq_len), emb_dim)])
        labels = self.data.get_labels(idx)
        item_dict = {
            "embeddings": embeddings,
            "3state": torch.cat([labels[0], torch.zeros((self.data.max_length - seq_len), 3)]),
            "8state": torch.cat([labels[1], torch.zeros((self.data.max_length - seq_len), 8)]),
            "2state": torch.cat([labels[2], torch.zeros((self.data.max_length - seq_len), 2)]),
            "seq_len": seq_len
            }
        return item_dict


class get_data(object):
    def __init__(self, datafile, model, max_seq_length, emb_batch_size=None):
        self.datafile = datafile
        self.model = model
        self.hiddens = []
        self.masks = []
        self.labels = []
        self.labels_str = []
        self.input_seq = []
        self.max_length = max_seq_length
        if model is not None:
            self.prot_model_batch_size = model.cfg.model.micro_batch_size
            self.emb_batch_size = emb_batch_size
        self._get_data()

    def _get_data(self):
        with torch.no_grad():
            with open(self.datafile, "r") as f:
                next(f) # skip  the header
                for _, line in tqdm(enumerate(f)):
                    line = line.split(",")
                    sequence = line[1].strip()
                    if len(sequence) > self.max_length - 2:
                        continue
                    self.input_seq.append(sequence)
                    numpy_threestate_labels=np.zeros((len(sequence), 3))
                    numpy_eightstate_labels=np.zeros((len(sequence), 8))
                    numpy_diso_labels=np.zeros((len(sequence), 2))
                    for idx, val in enumerate(line[2]):
                        label2num = three_state_label2num[val]
                        numpy_threestate_labels[idx][label2num] = 1
                    for idx, val in enumerate(line[3]):
                        label2num = eight_state_label2num[val]
                        numpy_eightstate_labels[idx][label2num] = 1
                    for idx, val in enumerate(line[4].strip()):
                        numpy_diso_labels[idx][int(val)] = 1

                    label = [torch.tensor(numpy_threestate_labels),
                            torch.tensor(numpy_eightstate_labels),
                            torch.tensor(numpy_diso_labels)]

                    label_str = [line[2], line[3], line[4].strip()]
                    self.labels.append(label)
                    self.labels_str.append(label_str)
        
        if self.model is not None:
            with torch.no_grad():
                _reconfigure_inference_batch(self.emb_batch_size) 

                batches = list(range(0, len(self.labels), self.emb_batch_size))
                if batches[-1] < len(self.labels):
                    batches.append(len(self.labels))
                for i in trange(len(batches) - 1):
                    hiddens, masks = self.compute_hiddens(self.input_seq[batches[i]:batches[i+1]])
                    self.hiddens += list(hiddens.cpu().float())
                    self.masks += list(masks.cpu().float())
                _reconfigure_inference_batch(self.prot_model_batch_size, global_batch_size=self.model.cfg.model.global_batch_size)

    def compute_hiddens(self, seqs):
        if self.model is not None:
            hiddens, masks = self.model.seq_to_hiddens(seqs)
            return hiddens, masks 
        else:
            return None

    def get_hidden_size(self):
        return self.model.cfg.model.hidden_size

    def get_embeddings(self, idx):
        if self.model is not None:
            return torch.squeeze(self.hiddens[idx][self.masks[idx].bool()])
        else:
            return self.input_seq[idx]

    def get_labels(self, idx):
        return self.labels[idx]
    
    def get_labels_str(self, idx):
        return self.labels_str[idx]

    def length(self):
        return len(self.labels)

    @staticmethod
    def tensor2list(tensor):
        """Convert multi-dimension tensor back to a single dimension list for labels"""
        labels = []
        #The tensor might be output from model in which case, it would be non-zero numbers.
        #we take the max value for every row and do one-hot encoding of that value as 1 and
        # rest as 0.
        onehot_tensor = torch.zeros_like(tensor).scatter(2, tensor.argmax(2,True), value=1)
        idxlist = ((onehot_tensor == 1).nonzero(as_tuple=True)[2]).tolist()
        return idxlist

    @staticmethod
    def num2label(sequence, state):
        num2label = []
        seq_list = get_data.tensor2list(sequence)
        if state == "three_state":
            converter = three_state_num2label
        elif state == "eight_state":
            converter = eight_state_num2label
        else:
            raise ValueError("Unknown state to convert to labels: %s".format(state))
        for num in seq_list:
            num2label.append(converter[num])
        return num2label
