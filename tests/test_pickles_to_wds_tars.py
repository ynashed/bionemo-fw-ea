# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import os
import pickle
import tarfile

from bionemo.data.diffdock.webdataset_utils import pickles_to_tars


BIONEMO_HOME = os.environ.get("BIONEMO_HOME", "/workspace/bionemo")
SOURCE_DATA = os.path.join(
    BIONEMO_HOME,
    "examples/tests/test_data/molecule/diffdock/data_cache/",
    "torsion_limit0_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings_INDEXsplit_train",
)
REF_TARFILE_SIZE = [27801600, 27463680, 27166720, 25139200]


def test_pickles_to_wds_tars(tmp_path):
    pickle_file_path = os.path.join(tmp_path, "pickle_files")
    os.makedirs(pickle_file_path)

    for file in glob.glob(os.path.join(SOURCE_DATA, "heterographs-*.tar")):
        tar = tarfile.open(file)
        tar.extractall(path=pickle_file_path)

    complex_names = [file.split(".")[0] for file in sorted(os.listdir(pickle_file_path))]
    assert len(complex_names) == 40

    output_path = os.path.join(tmp_path, "webdataset_tarfiles")
    os.makedirs(output_path)

    pickles_to_tars(
        pickle_file_path,
        "HeteroData.pyd",
        complex_names,
        output_path,
        "heterographs",
        lambda complex_graph: {"__key__": complex_graph.name, "HeteroData.pyd": pickle.dumps(complex_graph)},
        4,
    )
    assert len(os.listdir(output_path)) == 4

    for idx, file in enumerate(sorted(glob.glob(os.path.join(output_path, "heterographs-*.tar")))):
        assert os.stat(file).st_size == REF_TARFILE_SIZE[idx]
