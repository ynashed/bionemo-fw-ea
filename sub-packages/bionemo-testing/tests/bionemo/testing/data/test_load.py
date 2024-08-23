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


import gzip
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bionemo.testing.data.load import load
from bionemo.testing.data.resource import get_all_resources


@pytest.fixture()
def resources(tmp_path):
    (tmp_path / "foo.yaml").write_text(
        """
        - tag: "bar"
          pbss: "s3://test/bar"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        - tag: "baz"
          pbss: "s3://test/baz.gz"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        - tag: "dir"
          pbss: "s3://test/dir.tar"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        - tag: "dir.gz"
          pbss: "s3://test/dir.tar.gz"
          owner: Peter St John <pstjohn@nvidia.com>
          sha256: null
        """
    )

    return get_all_resources(tmp_path)


def test_load_raises_error_on_invalid_tag(resources):
    with pytest.raises(ValueError, match="Resource 'invalid/tag' not found."):
        load("invalid/tag", resources=resources)


def test_load_raises_with_invalid_source(resources):
    with pytest.raises(ValueError, match="Source 'invalid' not supported."):
        load("foo/bar", source="invalid", resources=resources)  # type: ignore


def test_load_raises_with_no_ngc_url(resources):
    with pytest.raises(ValueError, match="Resource 'foo/bar' does not have an NGC URL."):
        load("foo/bar", source="ngc", resources=resources)  # type: ignore


@patch("bionemo.testing.data.load._s3_download")
def test_load_with_file(mocked_s3_download, resources):
    mocked_s3_download.side_effect = lambda _1, output_file, _2: Path(output_file).write_text("test")
    file_path = load("foo/bar", resources=resources)
    assert file_path.is_file()
    assert file_path.read_text() == "test"


@patch("bionemo.testing.data.load._s3_download")
def test_load_with_gzipped_file(mocked_s3_download, resources):
    def write_compressed_text(_1, output_file: str, _2):
        with gzip.open(output_file, "wt") as f:
            f.write("test")

    mocked_s3_download.side_effect = write_compressed_text

    file_path = load("foo/baz", resources=resources)
    assert file_path.is_file()
    assert file_path.read_text() == "test"


@patch("bionemo.testing.data.load._s3_download")
def test_load_with_tar_directory(mocked_s3_download, resources):
    def write_compressed_dir(_1, output_file: str, _2):
        # Create a text file in memory
        text_content = "test"
        text_file = io.BytesIO(text_content.encode("utf-8"))

        # Create a tarfile
        with tarfile.open(output_file, "w") as tar:
            # Create a TarInfo object for the file
            info = tarfile.TarInfo(name="test_file")
            info.size = len(text_content)

            # Add the file to the tarfile
            tar.addfile(info, text_file)

    mocked_s3_download.side_effect = write_compressed_dir

    file_path = load("foo/dir", resources=resources)
    assert file_path.is_dir()
    assert (file_path / "test_file").read_text() == "test"


@patch("bionemo.testing.data.load._s3_download")
def test_load_with_targz_directory(mocked_s3_download, resources):
    def write_compressed_dir(_1, output_file: str, _2):
        # Create a text file in memory
        text_content = "test"
        text_file = io.BytesIO(text_content.encode("utf-8"))

        # Create a tarfile
        with tarfile.open(output_file, "w") as tar:
            # Create a TarInfo object for the file
            info = tarfile.TarInfo(name="test_file")
            info.size = len(text_content)

            # Add the file to the tarfile
            tar.addfile(info, text_file)

    mocked_s3_download.side_effect = write_compressed_dir

    file_path = load("foo/dir.gz", resources=resources)
    assert file_path.is_dir()
    assert (file_path / "test_file").read_text() == "test"
