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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import shutil
from pathlib import Path
from typing import Sequence

import requests


__all__: Sequence[str] = (
    "checked_download",
    "move",
    "copy",
)


def _copy_file(
    src: str,
    dest: str,
    create_nonexistent_paths: bool = True,
    overwrite: bool = True,
    destroy_src_on_copy: bool = False,
) -> None:
    assert os.path.exists(src)
    assert os.path.isfile(src)
    assert os.path.isfile(dest) or os.path.isdir(dest)

    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(f"Destination {dest} exists and cannot be overwritten.")
    elif not os.path.exists(dest):
        if create_nonexistent_paths:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory {dest} does not exist.")

    shutil.copy(src, dest)

    if destroy_src_on_copy:
        shutil.rmtree(src)


def move(src: str, dest: str, replace: bool = True) -> None:
    if not replace and os.path.isfile(dest) and os.path.exists(dest):
        raise FileExistsError(f"Destination exists, cowardly refusing to replace: {dest}")
    os.replace(src, dest)


def copy(
    src: str,
    dest: str,
    create_nonexistent_paths: bool = True,
    overwrite: bool = True,
    destroy_src_on_copy: bool = False,
) -> None:
    """
    Copies a file or directory to a new location.
    """
    # TODO Why does this function exist? Why not use shutil.copy(..) ??
    assert os.path.exists(src)

    if os.path.isfile(src):
        _copy_file(
            src,
            dest,
            create_nonexistent_paths=create_nonexistent_paths,
            overwrite=overwrite,
            destroy_src_on_copy=destroy_src_on_copy,
        )


def checked_download(local_path: str, remote_path: str) -> None:
    if os.path.exists(local_path):
        print(f"File exists locally: {local_path}")
        return

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading file {remote_path} to {local_path}")
        response = requests.get(remote_path)
        with open(local_path, "wb") as ofi:
            ofi.write(response.content)
    except requests.RequestException as e:
        print(f"Failed to download '{remote_path}': {e}")
