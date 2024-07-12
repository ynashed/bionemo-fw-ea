# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import importlib
from pathlib import Path

import pytest

import bionemo


package_path = bionemo.__file__.replace("__init__.py", "")

imports = []
for path in Path(package_path).rglob("*.py"):
    import_str = (
        str(path)
        .replace(package_path, "bionemo.")
        .replace("__init__.py", "")
        .replace(".py", "")
        .replace("/", ".")
        .strip(".")
    )
    imports.append(import_str)


@pytest.mark.parametrize("import_str", imports)
def test_import(import_str):
    print(import_str)
    try:
        importlib.import_module(import_str)
        assert True
    except Exception as e:
        print(e)
        assert False
