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


import numpy as np

from bionemo.data.scdl.io.sc_mmap_dataset import INT_WIDTH, MODE, SC_MMAP_Dataset, _swap_mmap_array


CURRENT_VERSION = "0.0.1"


def create_and_set(data, filepath):
    arr = np.memmap(f"{filepath}", dtype="uint32", shape=(len(data),), mode="w+")
    arr[0 : len(data)] = np.array(data, dtype="uint32")
    return arr


def test_accessory_functions(tmpdir):
    x = [1, 2, 3, 4, 5]
    x_arr = create_and_set(x, f"{tmpdir}/x.npy")

    y = [10, 9, 8, 7, 6, 5, 4, 3]
    y_arr = create_and_set(y, f"{tmpdir}/y.npy")

    _swap_mmap_array(x_arr, f"{tmpdir}/x.npy", y_arr, f"{tmpdir}/y.npy")
    x_now_y = np.memmap(f"{tmpdir}/y.npy", dtype="uint32", shape=(len(x),), mode="r+")
    y_now_x = np.memmap(f"{tmpdir}/x.npy", dtype="uint32", shape=(len(y),), mode="r+")
    assert len(x_now_y) == len(x)
    assert np.array_equal(x_now_y, np.array(x))
    assert len(y_now_x) == len(y)


def test_settings():
    assert INT_WIDTH.EIGHT == "8"
    assert INT_WIDTH.SIXTEEN == "16"

    assert MODE.CREATE_APPEND == "w+"
    assert MODE.READ_APPEND == "r+"


def test_object_creation(tmpdir):
    SC_MMAP_Dataset(f"{tmpdir}/scx")
    assert True


def test_load_hdf5(tmpdir):
    ds = SC_MMAP_Dataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    assert ds.n_obs() == 25382
    assert ds.n_vars() == 34455
    assert len(ds) == 25382
    assert ds.n_values() == 874536810
    assert ds.num_nonzeros() == 26947275
    assert ds.sparsity() == 0.9691868030117566
    assert len(ds) == 25382
    assert ds.version() == CURRENT_VERSION


def test_load_scmmap(tmpdir):
    ds = SC_MMAP_Dataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    ds.save()
    del ds
    reloaded = SC_MMAP_Dataset(f"{tmpdir}/scy")

    assert reloaded.n_obs() == 25382
    assert reloaded.n_vars() == 34455
    assert reloaded.n_values() == 874536810
    assert reloaded.num_nonzeros() == 26947275
    assert reloaded.sparsity() == 0.9691868030117566
    assert len(reloaded) == 25382
    assert len(reloaded[10][0]) == 1106
    vals, cols = reloaded[10]
    assert vals[0] == 3.0
    assert vals[1] == 1.0
    assert vals[10] == 25.0
    assert cols[10] == 488
    assert len(reloaded._get_row(10, return_features=False, pad=True)) == 34455

    padded_row, feats = reloaded._get_row(10, True, True, "feature_name", False)
    assert len(padded_row) == 34455
    assert padded_row[488] == 25.0
    assert len(feats) == 34455

    # ## TODO: copy construct not implemented
    # ds = SC_MMAP_Dataset("scz")
    # ds.load(f"{tmpdir}/scy")
    # assert reloaded.n_obs() == 25382
    # assert reloaded.n_vars() == 34455
    # assert reloaded.n_values() == 874536810
    # assert reloaded.num_nonzeros() == 26947275
    # assert reloaded.sparsity() == 0.9691868030117566
    # assert len(reloaded) == 25382

    ## Value retrieval
    # adata = ad.read_h5ad('hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad')
    # adata.X[300,450] = 1.4055748
    # adata.raw.X[300,450] = 1.0
    assert reloaded._get_row(300, return_features=False, pad=True)[450] == 1.0
    assert reloaded.get(300, 450) == 1.0
    assert reloaded.data[50000] == 1.0
    assert reloaded.data[55001] == 33.0
    ## adata.X[1985,1090]
    assert reloaded.get(1985, 1090) == 0.0
    assert reloaded.get(1985, 1090, False) is None
    assert reloaded.get(0, 488) == 15.0
    assert reloaded.get(25381, 32431) == 7.0
    assert reloaded.shape() == (25382, 34455)

    ##TODO: test getidx function
    # assert reloaded[25381][32431] == 7.0


def test_concat_mmaps_same(tmpdir):
    ds = SC_MMAP_Dataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    dt = SC_MMAP_Dataset(f"{tmpdir}/sct", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")

    dt.concat(ds)
    assert dt.n_obs() == 2 * ds.n_obs()
    assert dt.n_values() == 2 * ds.n_values()
    assert dt.num_nonzeros() == 2 * ds.num_nonzeros()


def test_concat_mmaps_diff(tmpdir):
    ds = SC_MMAP_Dataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    dt = SC_MMAP_Dataset(f"{tmpdir}/sct", h5ad_path="hdf5/5315d127-d698-44c5-955d-5e5c87e39ac3.h5ad")

    exp_n_obs = ds.n_obs() + dt.n_obs()
    exp_n_val = ds.n_values() + dt.n_values()
    exp_nnz = ds.num_nonzeros() + dt.num_nonzeros()
    dt.concat(ds)
    assert dt.n_obs() == exp_n_obs
    assert dt.n_values() == exp_n_val
    assert dt.num_nonzeros() == exp_nnz


def test_concat_mmaps_multi(tmpdir):
    ds = SC_MMAP_Dataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    dt = SC_MMAP_Dataset(f"{tmpdir}/sct", h5ad_path="hdf5/5315d127-d698-44c5-955d-5e5c87e39ac3.h5ad")
    dx = SC_MMAP_Dataset(f"{tmpdir}/sccx", h5ad_path="hdf5/sub/f8f41e86-e9ed-4de7-a155-836b2f243fd0.h5ad")
    exp_n_obs = ds.n_obs() + dt.n_obs() + dx.n_obs()
    dt.concat(ds)
    dt.concat(dx)

    assert dt.n_obs() == exp_n_obs

    dns = SC_MMAP_Dataset(f"{tmpdir}/scdns", h5ad_path="hdf5/5315d127-d698-44c5-955d-5e5c87e39ac3.h5ad")
    dns.concat([ds, dx])
    assert dns.n_obs() == dt.n_obs()
    assert dns.n_values() == dt.n_values()
    assert dns.num_nonzeros() == dt.num_nonzeros()
    assert dns.n_vars() == dt.n_vars()

    ## TODO: check specific values of the file.
