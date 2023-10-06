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

import logging
from concurrent import futures

import esm1nv_pb2_grpc
import grpc
from esm1nv_pb2 import OutputSpec
from hydra import compose, initialize

from bionemo.model.protein.esm1nv import ESM1nvInference


logger = logging.getLogger(__name__)


class InferenceService(esm1nv_pb2_grpc.GenerativeSampler):
    def __init__(self):
        if not hasattr(self, '_inferer'):
            with initialize(config_path="../../../../../examples/protein/esm1nv/conf"):
                inf_cfg = compose(config_name="infer")
                self._inferer = ESM1nvInference(cfg=inf_cfg)

    def SeqToEmbedding(self, spec, context):
        embeddings, masks = self._inferer.seq_to_hiddens(spec.seqs)
        embeddings[~masks.bool()] = 0
        embeddings = embeddings[:, masks.sum(axis=0) != 0, :]
        output = OutputSpec(embeddings=embeddings.flatten().tolist(), dim=embeddings.shape)
        return output


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    esm1nv_pb2_grpc.add_GenerativeSamplerServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f'[::]:{50051}')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
