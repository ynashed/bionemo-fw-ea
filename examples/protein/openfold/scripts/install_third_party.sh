#!/bin/bash
set -e

if [ -z "$BIONEMO_HOME" ]; then
    echo "\$BIONEMO_HOME is unset. Please set the variable and run the script again. This variable should be set to the base of the repo path."
    exit 1
fi

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
wget -q -P ${BIONEMO_HOME}/downloads https://github.com/soedinglab/hh-suite/archive/refs/tags/v3.3.0.tar.gz
tar -xzf ${BIONEMO_HOME}/downloads/v3.3.0.tar.gz --directory ${BIONEMO_HOME}
rm -r ${BIONEMO_HOME}/downloads
ls ${BIONEMO_HOME}
cd ${BIONEMO_HOME}/hh-suite-3.3.0
mkdir -p build
cd build
cmake ..
make -j
make install
rm -r ${BIONEMO_HOME}/hh-suite-3.3.0
echo "HH-suite v.3.3.0 Kalign v3.3.5 installed successfuly"
