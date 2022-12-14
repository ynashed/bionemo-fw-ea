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

# TODO: check if the following missing dependencies are needed
#   - lcms2=2.12=hddcbb42_0
#   - libgcc-ng=11.1.0=hc902ee8_8
#   - libstdcxx-ng=11.1.0=h56837e0_8
#   - pycairo=1.20.1=py38hf61ee4a_0


# TODO: update to nemo's official release when 1.14 is ready
# ARG BASE_IMAGE=nvcr.io/nvidia/nemo:22.07
ARG BASE_IMAGE=gitlab-master.nvidia.com/mlivne/nemo_containers:nemo-r1.14.0
FROM ${BASE_IMAGE}

ARG GITHUB_ACCESS_TOKEN
ARG GITHUB_BRANCH
ARG GITHUB_REPO=gitlab-master.nvidia.com/clara-discovery/bionemo.git
# The location of the installed library in the container. Set BIONEMO_PATH env to change this for development purposes.
ARG BIONEMO_HOME=/opt/nvidia/bionemo 
ARG NEMO_HOME=/opt/nvidia/nemo
ARG NEMO_BRANCH
ARG PACKAGE=0

# Activating base environment
ENV TERM=xterm

## Installing dependencies and tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3-setuptools python3-dev swig \
        wget git unzip tmux vim tree && \
    rm -rf /var/lib/apt/lists/*

COPY setup/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

## PySMILES
RUN git clone https://github.com/MolecularAI/pysmilesutils.git --branch master /opt/pysmilesutils \
    && cd /opt/pysmilesutils; pip install .

RUN git clone --branch ${GITHUB_BRANCH} \
    https://oauth2:${GITHUB_ACCESS_TOKEN}@${GITHUB_REPO} ${BIONEMO_HOME}

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
