# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# TODO: check if the following missing dependencies are needed
#   - lcms2=2.12=hddcbb42_0
#   - libgcc-ng=11.1.0=hc902ee8_8
#   - libstdcxx-ng=11.1.0=h56837e0_8
#   - pycairo=1.20.1=py38hf61ee4a_0

ARG BASE_IMAGE=nvcr.io/nvidia/nemo:23.10
FROM ${BASE_IMAGE}

ENV TERM=xterm
ENV FORCE_CUDA=1

# makes the interactive bash shell "login" have this in the prompt
RUN echo 'export PS1="(bionemo docker) $PS1"' > /docker_bashrc

WORKDIR /workspace/
# TODO: Long term, docker should not be running as root.
ENV HOME=/root
# copy built DGL and install it
# dgl_builder is created with setup/Dockerfile.dgl_builder
COPY --from=gitlab-master.nvidia.com:5005/clara-discovery/bionemo:dgl_builder_v1.1.3 /dgl ./dgl
RUN cd dgl/python && python setup.py install && cd ../.. && rm -rf dgl

# Please populate the labels below if using a custom base container (ie not NeMo GA)
# For nightly NeMo builds, only com.nvidia.nemo.description should be provided
# (release and git_hash labels are filled automatically).
# LABEL "com.nvidia.nemo.release"="r1.19.0"
# LABEL "com.nvidia.nemo.git_hash"="a5fbsegy"
# LABEL "com.nvidia.nemo.description"="Cusotom build with XYZ fixed"

## Installing dependencies and tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3-setuptools python3-dev swig \
        wget git unzip tmux vim emacs-nox tree xterm && \
    rm -rf /var/lib/apt/lists/*

# Optional: Install code-server to enable easy remote development on a container
# More info about code-server be found here: https://coder.com/docs/code-server/v4.4.0
# ADD https://github.com/coder/code-server/releases/download/v4.4.0/code-server_4.4.0_amd64.deb code-server_4.4.0_amd64.deb
# RUN dpkg -i ./code-server_4.4.0_amd64.deb && rm -f code-server_4.4.0_amd64.deb
#
# # Install extensions from the marketplace
# RUN code-server --install-extension ms-python.python


COPY setup/requirements.txt /tmp/requirements.txt
RUN --mount=type=secret,id=GITLAB_TOKEN GITLAB_TOKEN=$(cat /run/secrets/GITLAB_TOKEN) pip install -r /tmp/requirements.txt
# TODO: unnecessary for main container, move test+dev dep installation to dev only & update CI to use dev image
COPY setup/requirements-test.txt /tmp/requirements-test.txt
RUN pip install -r /tmp/requirements-test.txt
COPY setup/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install -r /tmp/requirements-dev.txt
# end TODO

# AWS CLI is inside requirements.txt. Add config file.
# This config file does not contain access keys.
RUN mkdir $HOME/.aws
COPY setup/aws-config-swiftstack $HOME/.aws/config

## Install NGC cli in container for artifact download
ENV NGC_CLI_INSTALL_PREFIX=/usr/local/bin
RUN wget -O /tmp/ngccli_linux.zip --content-disposition \
    "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.38.0/files/ngccli_linux.zip" && \
    unzip /tmp/ngccli_linux.zip -d ${NGC_CLI_INSTALL_PREFIX} && \
    chmod u+x ${NGC_CLI_INSTALL_PREFIX}/ngc-cli/ngc && \
    ln -fns  ${NGC_CLI_INSTALL_PREFIX}/ngc-cli/ngc ${NGC_CLI_INSTALL_PREFIX}/ngc && \
    rm /tmp/ngccli_linux.zip

# Recompile Megatron helpers -- NOTE this may need to be done before running job later
RUN NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])') && \
   cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron && \
   make

#########################################################################################
# Apply NeMo patches for key upstream changes between releases
#  TODO whenever we upgrade nemo see which can be removed.
#########################################################################################
# 1. Setting the workdir to the base nemo directory for all nemo patches

# NOTE: This is to support the MegaMolBart LM model compatability with Jupyter notebooks and ipdb.
#       This is exposed through the `iteractive=True` constructor parameter.
#       In MegaMolBart, specifically, this patch adds support for recongifure_minibatch as a flag
#       in its decode method.
# TODO [mgreaves] Upstream this patch and then remove here when we can upgrade our nemo dependency.
#                 https://jirasw.nvidia.com/browse/CDISCOVERY-2146
COPY ./setup/nemo-v-1.22-support_interactive_megamolbart_lm-decode_reconfigure_minibatch.patch ./
RUN NEMO_INST_DIR=$(python -c 'import nemo; from pathlib import Path; print(Path(nemo.__path__[0]).parent)') && \
    patch -p1 -d $NEMO_INST_DIR -i $PWD/nemo-v-1.22-support_interactive_megamolbart_lm-decode_reconfigure_minibatch.patch && \
    rm nemo-v-1.22-support_interactive_megamolbart_lm-decode_reconfigure_minibatch.patch

# NOTE: hiddens_support_train_v_inf_outputs.patch applies the patch from https://github.com/NVIDIA/NeMo/pull/8466 which fixes an issue
#        where training uses the same hiddens key for inference (eg always the gaussian noised z rather than z_mean).
COPY ./setup/hiddens_support_train_v_inf_outputs.patch ./
RUN NEMO_INST_DIR=$(python -c 'import nemo; from pathlib import Path; print(Path(nemo.__path__[0]).parent)') && \
    patch -p1 -d $NEMO_INST_DIR -i $PWD/hiddens_support_train_v_inf_outputs.patch && \
    rm hiddens_support_train_v_inf_outputs.patch

# NOTE: hiddens_support_train_v_inf_outputs.patch applies the patch from https://github.com/NVIDIA/NeMo/pull/8576 which fixes an issue
#        where you restart a run after an epoch.
COPY ./setup/data_sampler_constant_len.patch ./
RUN NEMO_INST_DIR=$(python -c 'import nemo; from pathlib import Path; print(Path(nemo.__path__[0]).parent)') && \
    patch -p1 -d $NEMO_INST_DIR -i $PWD/data_sampler_constant_len.patch && \
    rm data_sampler_constant_len.patch


#########################################################################################
# BioNeMo Build
#########################################################################################

### The location of the installed library in the container. Set BIONEMO_HOME env to change this for development purposes.
ARG BIONEMO_HOME=/workspace/bionemo
ENV BIONEMO_HOME=${BIONEMO_HOME}
RUN mkdir -p ${BIONEMO_HOME}
WORKDIR ${BIONEMO_HOME}
### Copy the bionemo repo contents for distribution
COPY LICENSE README.md artifact_paths.yaml download_models.py launch.sh setup.py pyproject.toml ${BIONEMO_HOME}/
COPY setup ${BIONEMO_HOME}/setup
COPY docs ${BIONEMO_HOME}/docs
COPY hydra_plugins ${BIONEMO_HOME}/hydra_plugins
COPY tokenizers ${BIONEMO_HOME}/tokenizers
# copy the package contents *last* as most churn occurs here
# --> we can cache the lower-chrun things in previous layers
COPY examples ${BIONEMO_HOME}/examples
COPY tests ${BIONEMO_HOME}/tests
COPY bionemo ${BIONEMO_HOME}/bionemo


### install bionemo
WORKDIR ${BIONEMO_HOME}
# we already install dependencies earlier, directly from the requirements*.txt files
RUN pip install --no-deps -e .

# Remove nvm as it's not expected users will need this, and it introduces unfixable vulnerabilties:
# https://nvd.nist.gov/vuln/detail/CVE-2023-32002
RUN /bin/bash -c "source /usr/local/nvm/nvm.sh && nvm deactivate && nvm unload && sed -i '/NVM/d' /root/.bashrc"
RUN rm -rf /usr/local/nvm
RUN /bin/bash -c "sed -i '/NVM/d' /root/.bashrc && sed -i '/nvm.sh/d' /etc/bash.bashrc"
# Remove OpenCV: We do not use it and it adds a security vulnerability which no current fix:
# https://nvd.nist.gov/vuln/detail/CVE-2023-2617
RUN pip uninstall -y opencv-python opencv
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2 /usr/local/include/opencv4 /usr/local/share/opencv4 /usr/local/lib/cmake/opencv4 /usr/local/bin/opencv*
# Remove Slurm: The BioNeMo FW Container can certainly be deployed to Slurm clusters, however having Slurm
# installed *inside* the container does not typically have a use case. Slurm introduces vulnerabilities
# into our container.
RUN DEBIAN_FRONTEND=noninteractive apt remove -y libslurm37 libpmi2-0
# Remove fonttool: We do not use it and it adds a security vulnerability: https://github.com/advisories/GHSA-6673-4983-2vx5
RUN pip uninstall -y fonttools


# FIXME: If BIONEMO_HOME _is not_ /workspace/bionemo, then this ENTRYPOINT is invalid!
ENTRYPOINT ["/workspace/bionemo/setup/startup.sh"]
