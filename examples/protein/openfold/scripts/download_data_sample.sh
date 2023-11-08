#!/bin/bash

DATA_PATH=$PROJECT_MOUNT/examples/tests/test_data/

if ! [ -z "$1" ]
  then
    DATA_PATH=$1
  else
    echo Data will be extracted to $DATA_PATH. You can change location by providing argument to \
    this script:  download_data_sample.sh \<data_path\>
fi

ngc registry resource download-version nvidian/cvai_bnmo_trng/openfold:processed_sample
tar -xvf openfold_vprocessed_sample/openfold_sample_data.tar.gz -C $DATA_PATH
rm -r openfold_vprocessed_sample/
