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


# TODO switch to 23.03 when it's available (we need at least NeMo at commit aaa0ccacf)
ARG BASE_IMAGE=nvcr.io/nvidian/nemo-nightly-main:23-04-18
FROM ${BASE_IMAGE}

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
    && cd /opt/pysmilesutils && pip install .

# Recompile Megatron helpers -- NOTE this will need to be done if PYTHONPATH changes later
RUN NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])') && \
   cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron && \
   make

#########################################################################################
# BioNeMo Build
#########################################################################################

# The location of the installed library in the container. Set BIONEMO_HOME env to change this for development purposes.
ARG BIONEMO_HOME=/opt/nvidia/bionemo 
ENV BIONEMO_HOME=${BIONEMO_HOME}

# Clone code into repo
# Set REPO_INCLUDE to ./ to copy all code in, for example if building with CI 
# if the code should be cloned, set REPO_INCLUDE to any existing file, such as README.md
ARG REPO_INCLUDE=README.md
COPY ${REPO_INCLUDE} ${BIONEMO_HOME}

## If REPO_INCLUDE is not "./", clone repo using SSH outside of CI
ARG GITHUB_BRANCH=dev
ARG GITHUB_REPO=gitlab-master.nvidia.com:12051/clara-discovery/bionemo.git
RUN if [ ${REPO_INCLUDE} != "./" ] ; then mkdir -p -m 0700 ~/.ssh && ssh-keyscan ${GITHUB_REPO%%[:/]*} >> ~/.ssh/known_hosts ; else true ; fi
RUN --mount=type=ssh if [ ${REPO_INCLUDE} != "./" ] ; then rm -rf ${BIONEMO_HOME} && git clone --branch ${GITHUB_BRANCH} ssh://git@${GITHUB_REPO} ${BIONEMO_HOME} ; else true ; fi

ARG NEMO_HOME=/opt/nvidia/nemo
ENV NEMO_HOME=${NEMO_HOME}

ENV PYTHONPATH=${BIONEMO_HOME}:${BIONEMO_HOME}/generated:${NEMO_HOME}:$PYTHONPATH

COPY ./setup/startup.sh /opt/nvidia
COPY ./models /model

RUN mkdir -p /workspace/bionemo
RUN ln -s /opt/nvidia/bionemo/examples/ /workspace/bionemo/examples
WORKDIR /workspace

ENTRYPOINT ["/opt/nvidia/startup.sh"]
