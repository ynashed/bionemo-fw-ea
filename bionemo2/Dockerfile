# Use the specified base image
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /workspace/

ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

RUN git clone https://github.com/NVIDIA/Megatron-LM.git && \
  cd Megatron-LM && \
  git checkout a5534c8f3e2c49ad8ce486f5cba3408e14f5fcc2 && \
  pip install .

# RUN git clone --branch core_r0.6.0 https://github.com/NVIDIA/Megatron-LM.git /workspace/megatron-lm

RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout f058162b215791b15507bb542f22ccfde49c872d && \
  pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

# Transformer Engine 1.2.0
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git fetch origin da30634a6c9ccdbb6c587b6c93b1860e4b038204 && \
  git checkout FETCH_HEAD && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .


RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git checkout b6595cbae2226ff553b44ff2b66527738ea4bdf2 && \
    pip uninstall -y nemo_toolkit sacrebleu && \
    pip install -r requirements/requirements_lightning.txt && \
    pip install -r requirements/requirements_common.txt && \
    sed -i "/torch/d" requirements/requirements.txt && \
    sed -i "/triton/d" requirements/requirements.txt && \
    # sed -i "s/^megatron_core.*/megatron-core/" requirements/requirements_nlp.txt && \
    sed -i "/megatron_core/d" requirements/requirements_nlp.txt && \
    pip install -r requirements/requirements.txt && \
    pip install -r requirements/requirements_nlp.txt && \
    pip install ".[nlp]"

# RUN pip uninstall megatron_core
# RUN pip install megatron_core

# Install any additional dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/bionemo2

# install devtools and test dependencies
COPY ./requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY ./requirements-test.txt .
RUN pip install -r requirements-test.txt

#
# install dependencies of all bionemo namespace packages
#

# TODO: install reqs of feature package(s)
# --
# COPY ./sub-packages/bionemo-.../requirements.txt ./sub-packages/bionemo-.../requirements.txt
# RUN pip install -r ./sub-packages/bionemo-.../requirements.txt
# ...

COPY ./sub-packages/bionemo-fw/requirements.txt ./sub-packages/bionemo-fw/requirements.txt
RUN pip install -r ./sub-packages/bionemo-fw/requirements.txt

COPY ./sub-packages/bionemo-contrib/requirements.txt ./sub-packages/bionemo-contrib/requirements.txt
RUN pip install -r ./sub-packages/bionemo-contrib/requirements.txt

#
# install all bionemo namespaced code
#

# NOTE: We do not install from the `_requirements.txt` file, which contains each namespaced packages'
#       inter-dependent bionemo2 subpackage relationships. This is because we manually install each
#       subpackage's code independently. This speeds up the image building significantly.

# TODO install code of feature package(s)
# --
# WORKDIR /workspace/bionemo2/
# COPY ./sub-packages/bionemo-... ./sub-packages/bionemo-...
# WORKDIR /workspace/bionemo2/sub-packages/bionemo-...
# RUN pip install -r _requirements.txt
# RUN pip install --no-deps -e .
# ...

WORKDIR /workspace/bionemo2/
COPY ./sub-packages/bionemo-fw/ ./sub-packages/bionemo-fw/
WORKDIR /workspace/bionemo2/sub-packages/bionemo-fw/
#RUN pip install -r _requirements.txt
RUN pip install --no-deps -e .

WORKDIR /workspace/bionemo2/
COPY ./sub-packages/bionemo-contrib/ ./sub-packages/bionemo-contrib/
WORKDIR /workspace/bionemo2/sub-packages/bionemo-contrib/
#RUN pip install -r _requirements.txt
RUN pip install --no-deps -e .

# Copy the rest of the bionemo2 project
WORKDIR /workspace/
# COPY . /workspace/bionemo2/
WORKDIR /workspace/bionemo2/
