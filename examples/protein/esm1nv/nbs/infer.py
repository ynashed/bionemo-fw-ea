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

import grpc
import torch
import logging
from esm1nv_pb2_grpc import GenerativeSamplerStub
from esm1nv_pb2 import InputSpec

log = logging.getLogger(__name__)


class ESMInferenceWrapper():

    def __init__(self):
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = GenerativeSamplerStub(channel)

    def seq_to_embedding(self, seqs):
        spec = InputSpec(seqs=seqs)
        resp = self.stub.SeqToEmbedding(spec)

        embeddings = torch.FloatTensor(list(resp.embeddings))
        embeddings = torch.reshape(embeddings, tuple(resp.dim)).cuda()

        return embeddings
