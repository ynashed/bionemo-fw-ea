#!/bin/bash
set -e

echo "Installing Kalign v3.3.5"
wget -q -P /tmp/downloads https://github.com/TimoLassmann/kalign/archive/refs/tags/v3.3.5.tar.gz
tar -xzf /tmp/downloads/v3.3.5.tar.gz --directory /tmp
rm -r /tmp/downloads
ls /tmp
cd /tmp/kalign-3.3.5
mkdir -p build
cd build
cmake ..
make -j
make install
rm -r /tmp/kalign-3.3.5
echo "Kalign v3.3.5 installed successfuly"

echo "Installing HH-suite v.3.3.0"
wget -q -P /workspace/downloads https://github.com/soedinglab/hh-suite/archive/refs/tags/v3.3.0.tar.gz
tar -xzf /workspace/downloads/v3.3.0.tar.gz --directory /workspace
rm -r /workspace/downloads
ls /workspace
cd /workspace/hh-suite-3.3.0
mkdir -p build
cd build
cmake ..
make -j
make install
rm -r /workspace/hh-suite-3.3.0
echo "HH-suite v.3.3.0 Kalign v3.3.5 installed successfuly"