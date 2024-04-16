#!/usr/bin/env bash

set -euo pipefail

mk_tmp_dir() {
    # https://stackoverflow.com/a/10823731/362021
    local UUID=$(hexdump -n 16 -v -e '/1 "%02X"' /dev/urandom)
    local TEMP_INSTALL=
    mkdir "${TEMP_INSTALL}"
    echo "${TEMP_INSTALL}"
}

temp_install_space="$(mk_tmp_dir)"
trap "rm -rf ${temp_install_space}" EXIT
# cleanup of ${temp_install_space} will occur on script exit

pushd "${temp_install_space}"
git clone https://github.com/jnwatson/py-lmdb.git
pushd py-lmdb/
git checkout 57c692050b8d4f67ff7bcdec7acf38598de7c295 # tags/py-lmdb_1.4.1
LMDB_FORCE_SYSTEM=1 LMDB_FORCE_CFFI=1 python setup.py install sdist bdist_wheel
