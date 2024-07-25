ARG commit
FROM gitlab-master.nvidia.com:5005/clara-discovery/bionemo:${commit}

# NOTE: fw2nim comes from setup/requirements.txt and is already in the image!

# use dumb-init as PID 1 handler
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y dumb-init

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

ARG BIONEMO_HOME=/workspace/bionemo
ENV BIONEMO_HOME=${BIONEMO_HOME}
WORKDIR ${BIONEMO_HOME}

# remove fw2nim in bionemo
RUN rm -rf fw2nim/

# copy over MolMIM model
RUN mkdir -p ${BIONEMO_HOME}/models/molecule/molmim
COPY ./models/molmim_70m_24_3.nemo ${BIONEMO_HOME}/models/molmim_70m_24_3.nemo
RUN ln -s ${BIONEMO_HOME}/models/molmim_70m_24_3.nemo ${BIONEMO_HOME}/models/molecule/molmim/molmim_70m_24_3.nemo

# install molmim types
COPY ./fw2nim_examples/ ${BIONEMO_HOME}/fw2nim_examples/
WORKDIR ${BIONEMO_HOME}/fw2nim_examples/molmim
RUN pip install --no-deps -e .

WORKDIR ${BIONEMO_HOME}
CMD python -m fw2nim.exe.full_serving --full-config /workspace/bionemo/fw2nim_examples/molmim/fw_complete_config.yaml
