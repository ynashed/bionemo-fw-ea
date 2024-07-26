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

import pandas as pd
from scdl.index.FeatureIndex import RowFeatureIndex
from scdl.io.SCMMAPDataset import SC_MMAP_Dataset


def test_feature_index_internals():
    feats = pd.DataFrame({"feature_name": ["FF", "GG", "HH"], "feature_int": [1, 2, 3]})
    two_feats = pd.DataFrame(
        {
            "feature_name": ["FF", "GG", "HH", "II", "ZZ"],
            "gene_name": ["RET", "NTRK", "PPARG", "TSHR", "EGFR"],
            "spare": [None, None, None, None, None],
        }
    )
    index = RowFeatureIndex()
    assert isinstance(index.version(), str)
    assert len(index) == 0
    assert index.version() == "0.0.1"
    assert not index._index_ready
    assert index.n_rows() == 0

    index._update_index(12543, feats)
    assert len(index) == 1
    assert index._index_ready
    assert [3] == index.column_dims("feature_name")
    assert index.n_rows() == 12543

    ## Test n-values
    vals = index.n_values()
    assert vals == [12543 * 3]
    assert len(vals) == 1
    assert index.n_values(return_sum=True) == 12543 * 3

    index._update_index(455987, two_feats, "MY DATAFRAME")
    assert len(index) == 2
    assert index.n_vars_at_row(1, "feature_name") == 3
    assert index.n_vars_at_row(12544, "feature_name") == 5
    assert index.n_vars_at_row(455986) == 5
    assert index.n_vars_at_row(17, "feature_int") == 3
    assert index.n_values(return_sum=True) == (12543 * 3) + (455987 * 5)
    assert index.n_values()[1] == (455987 * 5)
    assert index.n_rows() == 455987 + 12543

    feats, label = index.lookup(3, None, True)
    assert list(feats.columns) == ["feature_name", "feature_int"]
    assert label is None

    feats, label = index.lookup(13001, None, True)
    assert label == "MY DATAFRAME"
    assert list(feats.columns) == list(two_feats.columns)


def test_index_with_SCMMAP(tmpdir):
    ds = SC_MMAP_Dataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    dt = SC_MMAP_Dataset(f"{tmpdir}/scx", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    feats = ds._feature_index.lookup(2, select_features=["feature_name"])
    assert len(feats) == 34455

    index = ds.features()
    index.index(dt)
    feats, label = index.lookup(25382 + 100, None, True)
    assert len(feats) == 34455
    assert isinstance(label, str)
    assert label == "hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad"
    assert len(index._feature_arr) == 2
    assert len(index._cumulative_sum_index) == 3

    index.save(f"{tmpdir}/test.idx")
    del index
    assert os.path.exists(f"{tmpdir}/test.idx")

    vindex = RowFeatureIndex.load(f"{tmpdir}/test.idx")
    feats, label = vindex.lookup(25382 + 100, None, True)
    assert len(feats) == 34455
    assert isinstance(label, str)
    assert label == "hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad"
    assert len(vindex._feature_arr) == 2
    assert len(vindex._cumulative_sum_index) == 3
    assert vindex._index_ready


def testindex_concat(tmpdir):
    ds = SC_MMAP_Dataset(f"{tmpdir}/scy", h5ad_path="hdf5/97e96fb1-8caf-4f08-9174-27308eabd4ea.h5ad")
    dt = SC_MMAP_Dataset(f"{tmpdir}/scx", h5ad_path="hdf5/sub/f8f41e86-e9ed-4de7-a155-836b2f243fd0.h5ad")
    index = RowFeatureIndex()
    index.index(ds)
    feats = index.lookup(2, select_features=["feature_name"])
    assert len(feats) == 34455

    ## Use concat, rather than .index
    dt_index = RowFeatureIndex()
    dt_index.index(dt)
    del dt

    index.concat(dt_index)
    del dt_index

    feats, label = index.lookup(25382 + 100, None, True)
    assert len(feats) == 36263
    assert isinstance(label, str)
    assert label == "hdf5/sub/f8f41e86-e9ed-4de7-a155-836b2f243fd0.h5ad"
    assert len(index._feature_arr) == 2
    assert len(index._cumulative_sum_index) == 3
    feats = index.lookup(2500)
    assert len(feats) == 34455
    assert index.n_vars_at_row(224) == 34455
