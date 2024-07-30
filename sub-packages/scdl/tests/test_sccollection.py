# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from bionemo.data.scdl.io.sc_collection import SingleCellCollection
from bionemo.data.scdl.io.sc_mmap_dataset import SingleCellMemMapDataset


def test_sccollection_internals(tmpdir):
    coll = SingleCellCollection(f"{tmpdir}/sccz")
    assert isinstance(coll.version(), str)
    assert coll.version() == "0.0.1"
    assert len(coll.mmap_paths) == 0
    assert coll.n_obs() == 0
    assert coll.n_vars() == 0
    assert coll.n_values() == 0
    assert coll.num_nonzeros() == 0
    assert coll.data_path == f"{tmpdir}/sccz"


def test_sccollection_basics(tmpdir):
    coll = SingleCellCollection(f"{tmpdir}/sccz")
    coll.load_h5ad("hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    assert coll.n_obs() == 25382
    assert coll.n_vars() == 34455
    assert coll.n_values() == 874536810
    assert coll.num_nonzeros() == 26947275
    assert coll.sparsity() == 0.9691868030117566
    assert coll.shape() == (25382, 34455)


def test_sccollection_multi(tmpdir):
    coll = SingleCellCollection(f"{tmpdir}/sccy")
    coll.load_h5ad_multi("hdf5/", max_workers=4, use_processes=False)
    assert len(coll.fname_to_mmap) == 3
    assert len(coll.mmap_paths) == 3
    assert coll.n_obs() == 96308
    assert sorted(coll.n_vars()) == [30480, 34455, 36263]
    assert coll.num_nonzeros() == 115136865
    assert coll.n_values() == 3423735545
    assert isinstance(coll.sparsity(), float)
    assert coll.sparsity() == 0.9663709817867957
    shape = coll.shape()
    assert isinstance(shape[0], int)
    assert isinstance(shape[1], list)
    assert shape[0] == 96308


def test_sccollection_serialization(tmpdir):
    coll = SingleCellCollection(f"{tmpdir}/sccy")
    coll.load_h5ad_multi("hdf5/", max_workers=4, use_processes=False)
    coll.flatten(f"{tmpdir}/flattened")
    dat = SingleCellMemMapDataset(f"{tmpdir}/flattened")
    assert dat.n_obs() == 96308
    assert dat.n_values() == 3423735545
    assert dat.num_nonzeros() == 115136865


def test_sccollection_concat(tmpdir):
    pass
