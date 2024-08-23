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
from typing import Annotated

import pydantic
import yaml


class Resource(pydantic.BaseModel):
    """Class that represents a remote resource for downloading and caching test data."""

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    tag: str
    """A unique identifier for the resource.

    The file(s) will be accessible via load("filename/tag"), where everything before the first slash is the name of the
    containing file (with the yaml suffix dropped), and everything after the first slash is the tag. Therefore if the
    tag contained a forward slash, the you could load the file with load("filename/tag/with/slash").
    """

    ngc: str | None = None
    """The NGC URL for the resource. If None, the resource is not available on NGC."""

    pbss: Annotated[pydantic.AnyUrl, pydantic.UrlConstraints(allowed_schemes=["s3"])]
    """The PBSS (NVIDIA-internal) URL of the resource."""

    sha256: str | None
    """The SHA256 checksum of the resource. If None, the SHA will not be checked on download (not recommended)."""

    owner: pydantic.NameEmail
    """The owner or primary point of contact for the resource, in the format "Name <email>"."""

    description: str | None = None
    """A description of the file(s)."""


def get_all_resources(resource_path: Path | None = None) -> dict[str, Resource]:
    """Return a dictionary of all resources."""
    if not resource_path:
        resource_path = Path(__file__).parent / "resources"

    resources_files = resource_path.glob("*.yaml")

    all_resources = []
    for file in resources_files:
        with file.open("r") as f:
            resources = yaml.safe_load(f)
            for resource in resources:
                resource["tag"] = f"{file.stem}/{resource['tag']}"
            all_resources += resources

    resource_list = pydantic.TypeAdapter(list[Resource]).validate_python(all_resources)
    resource_dict = {resource.tag: resource for resource in resource_list}

    if len(resource_dict) != len(resource_list):
        raise ValueError("Duplicate resource tags found!")

    return resource_dict
