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


from pathlib import Path

from bionemo.testing.data.resource import Resource, get_all_resources


def test_get_all_resources_returns_valid_entries():
    resources = get_all_resources()
    assert len(resources) > 0
    assert all(isinstance(resource, Resource) for resource in resources.values())


def test_get_all_resources_returns_combines_multiple_yamls(tmp_path: Path):
    (tmp_path / "resources1.yaml").write_text(
        """
        - tag: "foo"
          ngc: "bar"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        """
    )

    (tmp_path / "resources2.yaml").write_text(
        """
        - tag: "foo2"
          ngc: "bar"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        """
    )

    resources = get_all_resources(tmp_path)
    assert len(resources) == 2


def test_get_all_resources_returns_assigns_correct_tag(tmp_path: Path):
    (tmp_path / "file_name.yaml").write_text(
        """
        - tag: "tag_name"
          ngc: "bar"
          pbss: "s3://baz"
          sha256: "qux"
          owner: Peter St John <pstjohn@nvidia.com>
          description: "quux"
        """
    )

    resources = get_all_resources(tmp_path)
    assert "file_name/tag_name" in resources
