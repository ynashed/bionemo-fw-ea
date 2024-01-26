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
import pkgutil

import pytest


MODULE = 'dgl'


@pytest.mark.needs_gpu
def test_module_exist():
    """
    test to check if dgl exist
    """
    eggs_loader = pkgutil.find_loader(MODULE)
    assert eggs_loader is not None


@pytest.mark.needs_gpu
def test_module_import():
    """
    test to check if dgl can be safely imported
    """
    importlib.import_module(MODULE)
