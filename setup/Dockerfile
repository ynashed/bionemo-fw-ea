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

ARG BASE_IMAGE=nvcr.io/nvidia/nemo:22.07
FROM ${BASE_IMAGE}

ARG GITHUB_ACCESS_TOKEN
ARG GITHUB_BRANCH
ARG GITHUB_REPO=gitlab-master.nvidia.com/clara-discovery/bionemo.git
# The location of the installed library in the container. Set BIONEMO_PATH env to change this for development purposes.
ARG BIONEMO_HOME=/opt/nvidia/bionemo 
ARG NEMO_HOME=/opt/nvidia/nemo
ARG NEMO_BRANCH
ARG CONDA_ENV=base
ARG PACKAGE=0

# Activating base environment
RUN echo "source activate ${CONDA_ENV}" > /etc/bash.bashrc
ENV TERM=xterm

## Installing dependencies and tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python-setuptools python-dev swig \
        wget git unzip tmux vim tree && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pyfastx==0.8.4
COPY setup/conda/env.yml /tmp/env.yml
RUN conda env update --name base -f /tmp/env.yml && conda clean -afy

## PySMILES
RUN git clone https://github.com/MolecularAI/pysmilesutils.git --branch master /opt/pysmilesutils \
    && cd /opt/pysmilesutils; pip install .

RUN git clone --branch ${GITHUB_BRANCH} \
    https://oauth2:${GITHUB_ACCESS_TOKEN}@${GITHUB_REPO} ${BIONEMO_HOME}

RUN cd ${BIONEMO_HOME} && conda run -n ${CONDA_ENV} pip install -e .

# Recompile Megatron helpers -- NOTE this will need to be done if PYTHONPATH changes later
ENV PYTHONPATH=${BIONEMO_HOME}:${BIONEMO_HOME}/generated:${NEMO_HOME}:$PYTHONPATH
ENV PACKAGE=${PACKAGE}
ENV BIONEMO_HOME=${BIONEMO_HOME}
ENV NEMO_HOME=${NEMO_HOME}
ENV NEMO_BRANCH=${NEMO_BRANCH}

RUN NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])') && \
   cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron && \
   make

COPY ./setup/startup.sh /opt/nvidia
COPY ./models /model

RUN mkdir -p /workspace/bionemo
RUN ln -s /opt/nvidia/bionemo/examples/ /workspace/bionemo/examples
WORKDIR /workspace

ENTRYPOINT ["/opt/nvidia/startup.sh"]
