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


import contextlib
import os
from pathlib import Path
from typing import Literal

import boto3
import platformdirs
import pooch
from botocore.config import Config
from tqdm import tqdm

from bionemo.testing.data import resource


def _default_pbss_client():
    retry_config = Config(retries={"max_attempts": 10, "mode": "standard"})
    return boto3.client("s3", endpoint_url="https://pbss.s8k.io", config=retry_config)


def _get_cache_dir() -> Path:
    """Get the cache directory for downloaded resources."""
    if cache_dir := os.getenv("BIONEMO_CACHE_DIR"):
        return Path(cache_dir)

    return Path(platformdirs.user_cache_dir(appname="bionemo", appauthor="nvidia"))


BIONEMO_CACHE_DIR = _get_cache_dir()
BIONEMO_CACHE_DIR.mkdir(exist_ok=True)
RESOURCES = resource.get_all_resources()


def _s3_download(url: str, output_file: str | Path, _: pooch.Pooch) -> None:
    """Download a file from PBSS."""
    # Parse S3 URL to get bucket and key
    parts = url.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])

    with contextlib.closing(_default_pbss_client()) as s3:
        object_size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        progress_bar = tqdm(total=object_size, unit="B", unit_scale=True, desc=url)

        # Define callback
        def progress_callback(bytes_transferred):
            progress_bar.update(bytes_transferred)

        # Download file from S3
        s3.download_file(bucket, key, output_file, Callback=progress_callback)


def _ngc_download(url: str, output_file: str | Path, _: pooch.Pooch) -> None:
    raise NotImplementedError("NGC download not implemented.")


def load(
    model_or_data_tag: str,
    source: Literal["ngc", "pbss"] = "pbss",
    resources: dict[str, resource.Resource] | None = None,
) -> Path:
    """Download a resource from PBSS or NGC.

    Args:
        model_or_data_tag: A pointer to the desired resource. Must be a key in the resources dictionary.
        source: Either "pbss" (NVIDIA-internal download) or "ngc" (NVIDIA GPU Cloud). Defaults to "pbss".
        resources: A custom dictionary of resources. If None, the default resources will be used. (Mostly for testing.)

    Raises:
        ValueError: If the desired tag was not found, or if an NGC url was requested but not provided.

    Returns:
        A Path object pointing either at the downloaded file, or at a decompressed folder containing the
        file(s).

    Examples:
        For a resource specified in 'filename.yaml' with tag 'tag', the following will download the file:
        >>> load("filename/tag")
        PosixPath(/tmp/bionemo/downloaded-file-name)
    """
    if resources is None:
        resources = RESOURCES

    if model_or_data_tag not in resources:
        raise ValueError(f"Resource '{model_or_data_tag}' not found.")

    resource = resources[model_or_data_tag]

    match source:
        case "pbss":
            download_func = _s3_download
            url = resource.pbss
        case "ngc":
            download_func = _ngc_download
            url = resource.ngc
            if resource.ngc is None:
                raise ValueError(f"Resource '{model_or_data_tag}' does not have an NGC URL.")
        case _:
            raise ValueError(f"Source '{source}' not supported.")

    match "".join(Path(str(url)).suffixes):
        case ".gz" | ".bz2" | ".xz":
            processor = pooch.Decompress()

        case ".tar" | ".tar.gz":
            processor = pooch.Untar()

        case ".zip":
            processor = pooch.Unzip()

        case _:
            processor = None

    download = pooch.retrieve(
        url=str(url),
        known_hash=resource.sha256,
        path=BIONEMO_CACHE_DIR,
        downloader=download_func,
        processor=processor,
    )

    # Pooch by default returns a list of unpacked files if they unpack a zipped or tarred directory. Instead of that, we
    # just want the unpacked, parent folder.
    if isinstance(download, list):
        return Path(processor.extract_dir)  # type: ignore

    else:
        return Path(download)
