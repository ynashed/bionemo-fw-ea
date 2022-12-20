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

from nemo.utils import AppState, logging
from tqdm import tqdm, trange

from apex.transformer import parallel_state

from apex.transformer.pipeline_parallel.utils import (
    _reconfigure_microbatch_calculator,
)

three_state_label2num = {"C": 0, "H": 1, "E": 2}
eight_state_label2num = {"_": 0, "E": 1, "G": 2, "T": 3, "H": 4, "S": 5, "B": 6, "I": 7}

three_state_num2label = {0: "C", 1: "H", 2: "E"}
eight_state_num2label = {0: "_", 1: "E", 2: "G", 3: "T", 4: "H", 5: "S", 6: "B", 7: "I"}

# TODO: replace for similar function from model.utils
def reconfigure_microbatch(new_batch_size_per_gpu):
    app_state = AppState()
    from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=new_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
        micro_batch_size=new_batch_size_per_gpu,  # Make sure that there is no "grad acc" while decoding.
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
    )


class getData(object):
    def __init__(self, datafile, model_arch, model, emb_batch_size, prot_model_batch_size):
        self.datafile = datafile
        self.model = model
        self.model_arch = model_arch
        self.tokenizer = self.model.tokenizer
        self.embeddings = []
        self.labels = []
        self.input_seq = []
        self.labels_str = []
        self.max_length = self.model.cfg.model.seq_length
        self.emb_batch_size = emb_batch_size
        self.prot_model_batch_size = prot_model_batch_size
        self._get_data()

    # TODO: switch to a new inference interface
    def compute_embeddings(self, seqs):
        embeddings, _ = self.model._transform(seqs)
        return embeddings 

    def _get_data(self):
        
        with torch.no_grad():
            with open(self.datafile, "r") as f:
                next(f) # skip  the header
                for _, line in tqdm(enumerate(f)):
                    line = line.split(",")
                    sequence = line[1].strip()
                    if len(sequence) > self.model.cfg.model.seq_length - 2:
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
        
        with torch.no_grad():
            reconfigure_microbatch(self.emb_batch_size) 

            batches = list(range(0, len(self.labels), self.emb_batch_size))
            if batches[-1] < len(self.labels):
                batches.append(len(self.labels))
            for i in trange(len(batches) - 1):
                embeddings = self.compute_embeddings(self.input_seq[batches[i]:batches[i+1]])
                self.embeddings += list(embeddings.cpu()[:, 1:-1, :].float())
            
            reconfigure_microbatch(self.prot_model_batch_size)

    def get_hidden_size(self):
        return self.model.cfg.model.hidden_size


    def get_embeddings(self, idx):
        return torch.squeeze(self.embeddings[idx])

    def get_labels(self, idx):
        return self.labels[idx]
    
    def get_labels_str(self, idx):
        return self.labels_str[idx]

    def length(self):
        return len(self.labels)

    def get_input_seq(self, idx):
        return self.input_seq[idx]

    def tensor2list(self, tensor):
        """Convert multi-dimension tensor back to a single dimension list for labels"""
        labels = []
        #The tensor might be output from model in which case, it would be non-zero numbers.
        #we take the max value for every row and do one-hot encoding of that value as 1 and
        # rest as 0.
        onehot_tensor = torch.zeros_like(tensor).scatter(2, tensor.argmax(2,True), value=1)
        idxlist = ((onehot_tensor == 1).nonzero(as_tuple=True)[2]).tolist()
        return idxlist

    def num2label(self, sequence, state):
        num2label = []
        seq_list = self.tensor2list(sequence)
        if state == "three_state":
            converter = three_state_num2label
        elif state == "eight_state":
            converter = eight_state_num2label
        else:
            raise ValueError("Unknown state to convert to labels: %s".format(state))
        for num in seq_list:
            num2label.append(converter[num])
        return num2label

