#!/bin/bash

pip install --disable-pip-version-check --no-cache-dir -e 3rdparty/Megatron-LM
pip install --disable-pip-version-check --no-cache-dir -e 3rdparty/NeMo[all]
pip install --disable-pip-version-check --no-cache-dir -e sub-packages/bionemo-core
pip install --disable-pip-version-check --no-cache-dir -e sub-packages/bionemo-esm2
pip install --disable-pip-version-check --no-cache-dir -e sub-packages/bionemo-fw
pip install --disable-pip-version-check --no-cache-dir -e sub-packages/bionemo-geneformer
pip install --disable-pip-version-check --no-cache-dir -e sub-packages/bionemo-llm
pip install --disable-pip-version-check --no-cache-dir -e sub-packages/bionemo-testing
