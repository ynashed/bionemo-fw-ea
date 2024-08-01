#!/bin/bash

pip install --disable-pip-version-check --no-cache-dir -e 3rdparty/Megatron-LM
pip install --disable-pip-version-check --no-cache-dir -e 3rdparty/NeMo[all]
pip install --disable-pip-version-check --no-cache-dir -e sub-packages/bionemo-fw
pip install --disable-pip-version-check --no-cache-dir -e sub-packages/bionemo-contrib
