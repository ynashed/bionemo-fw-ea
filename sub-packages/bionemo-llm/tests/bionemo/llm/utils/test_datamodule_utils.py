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


from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size


def test_float_or_int_or_none_type_float():
    """
    Test that float_or_int_or_none returns a float when given a float on edge case 1.0
    """
    assert isinstance(float_or_int_or_none(1.0), float)
    assert isinstance(float_or_int_or_none("1.0"), float)


def test_float_or_int_or_none_type_int():
    """
    Test that float_or_int_or_none returns an int when given an int on edge case 1
    """
    assert isinstance(float_or_int_or_none(1), int)
    assert isinstance(float_or_int_or_none("1"), int)


def test_float_or_int_or_none_type_none():
    """
    Test that float_or_int_or_none returns None when given None
    """
    assert float_or_int_or_none(None) is None
    assert float_or_int_or_none("None") is None
