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

import os

import numpy as np
import pytest
import requests

from bionemo.data.scdl.io.sc_mmap_dataset import SingleCellMemMapDataset, _swap_mmap_array


CURRENT_VERSION = "0.0.1"


@pytest.fixture
def fn_download(tmpdir, request):  # tmpdir
    remote_path = f"https://datasets.cellxgene.cziscience.com/{request.param}"
    local_path = f"{tmpdir}/{request.param}"
    if not os.path.exists(local_path):
        response = requests.get(remote_path)
        with open(local_path, "wb") as ofi:
            ofi.write(response.content)
    yield local_path

    if os.path.exists(local_path):
        os.remove(local_path)


@pytest.mark.parametrize("fn_download", ["97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad"], indirect=True)
def test_load_hdf5(tmpdir, fn_download):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path=fn_download)
    assert ds.n_obs() == 25382
    assert ds.n_vars() == 34455
    assert len(ds) == 25382
    assert ds.n_values() == 874536810
    assert ds.num_nonzeros() == 26947275
    assert ds.sparsity() == 0.9691868030117566
    assert len(ds) == 25382
    assert ds.version() == CURRENT_VERSION


def create_and_set(data, file_name):
    arr = np.memmap(file_name, dtype="uint32", shape=(len(data),), mode="w+")
    arr[:] = np.array(data, dtype="uint32")
    return arr


def test_swap_mmap(tmpdir):
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
    assert np.array_equal(y_now_x, np.array(y))


def test_swap_mmap_no_file(tmpdir):
    x = [1, 2, 3, 4, 5]
    x_arr = create_and_set(x, f"{tmpdir}/x.npy")

    y = [10, 9, 8, 7, 6, 5, 4, 3]
    y_arr = create_and_set(y, f"{tmpdir}/y.npy")
    with pytest.raises(AssertionError):
        _swap_mmap_array(x_arr, f"{tmpdir}/x.npy", y_arr, f"{tmpdir}/z.npy")


def test_swap_mmap_delete(tmpdir):
    x = [1, 2, 3, 4, 5]
    x_arr = create_and_set(x, f"{tmpdir}/x.npy")

    y = [10, 9, 8, 7, 6, 5, 4, 3]
    y_arr = create_and_set(y, f"{tmpdir}/y.npy")
    _swap_mmap_array(x_arr, f"{tmpdir}/x.npy", y_arr, f"{tmpdir}/y.npy", destroy_src=True)
    assert not os.path.exists(f"{tmpdir}/x.npy")
    x_now_y = np.memmap(f"{tmpdir}/y.npy", dtype="uint32", shape=(len(x),), mode="r+")
    assert len(x_now_y) == len(x)
    assert np.array_equal(x_now_y, np.array(x))


@pytest.mark.parametrize("fn_download", ["97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad"], indirect=True)
def test_load_scmmap(tmpdir, fn_download):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path=fn_download)
    ds.save()
    del ds
    reloaded = SingleCellMemMapDataset(f"{tmpdir}/scy")

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
    # ds = SingleCellMemMapDataset("scz")
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


"""

def test_concat_mmaps_same(tmpdir):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    dt = SingleCellMemMapDataset(f"{tmpdir}/sct", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")

    dt.concat(ds)
    assert dt.n_obs() == 2 * ds.n_obs()
    assert dt.n_values() == 2 * ds.n_values()
    assert dt.num_nonzeros() == 2 * ds.num_nonzeros()


def test_concat_mmaps_diff(tmpdir):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    dt = SingleCellMemMapDataset(f"{tmpdir}/sct", h5ad_path="hdf5/5315d127-d698-44c5-955d-5e5c87e39ac3.h5ad")

    exp_n_obs = ds.n_obs() + dt.n_obs()
    exp_n_val = ds.n_values() + dt.n_values()
    exp_nnz = ds.num_nonzeros() + dt.num_nonzeros()
    dt.concat(ds)
    assert dt.n_obs() == exp_n_obs
    assert dt.n_values() == exp_n_val
    assert dt.num_nonzeros() == exp_nnz


def test_concat_mmaps_multi(tmpdir):
    ds = SingleCellMemMapDataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    dt = SingleCellMemMapDataset(f"{tmpdir}/sct", h5ad_path="hdf5/5315d127-d698-44c5-955d-5e5c87e39ac3.h5ad")
    dx = SingleCellMemMapDataset(f"{tmpdir}/sccx", h5ad_path="hdf5/sub/f8f41e86-e9ed-4de7-a155-836b2f243fd0.h5ad")
    exp_n_obs = ds.n_obs() + dt.n_obs() + dx.n_obs()
    dt.concat(ds)
    dt.concat(dx)

    assert dt.n_obs() == exp_n_obs

    dns = SingleCellMemMapDataset(f"{tmpdir}/scdns", h5ad_path="hdf5/5315d127-d698-44c5-955d-5e5c87e39ac3.h5ad")
    dns.concat([ds, dx])
    assert dns.n_obs() == dt.n_obs()
    assert dns.n_values() == dt.n_values()
    assert dns.num_nonzeros() == dt.num_nonzeros()
    assert dns.n_vars() == dt.n_vars()

    ## TODO: check specific values of the file.
"""
