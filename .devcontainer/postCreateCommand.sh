#!/bin/bash

pip install --disable-pip-version-check --no-cache-dir -e 3rdparty/Megatron-LM
pip install --disable-pip-version-check --no-cache-dir -e 3rdparty/NeMo[all]
for SUB_PKG in $(ls -d sub-packages/*)
do
    pip install --disable-pip-version-check --no-cache-dir -e $SUB_PKG
done
