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


import numba


@numba.njit(fastmath=True)
def permute(i: int, l: int, p: int) -> int:
    """Index into a permuted array with constant space and time complexity.

    This function permutes an index `i` into a range `[0, l)` using a hash function. See
    https://afnan.io/posts/2019-04-05-explaining-the-hashed-permutation/ for more details and
    "Correlated Multi-Jittered Sampling" by Andrew Kensler for the original algorithm.

    Args:
        i: The index to permute.
        l: The range of the permuted index.
        p: The permutation seed.

    Returns:
        The permuted index.
    """
    if l <= 0:
        raise ValueError("The range of the permuted index must be greater than 0.")

    if i not in range(l):
        raise ValueError("The index to permute must be in the range [0, l).")

    if p < 0:
        raise ValueError("The permutation seed must be greater than or equal to 0.")

    w = l - 1
    w |= w >> 1
    w |= w >> 2
    w |= w >> 4
    w |= w >> 8
    w |= w >> 16

    while True:
        i ^= p
        i *= 0xE170893D
        i ^= p >> 16
        i ^= (i & w) >> 4
        i ^= p >> 8
        i *= 0x0929EB3F
        i ^= p >> 23
        i ^= (i & w) >> 1
        i *= 1 | p >> 27
        i *= 0x6935FA69
        i ^= (i & w) >> 11
        i *= 0x74DCB303
        i ^= (i & w) >> 2
        i *= 0x9E501CC3
        i ^= (i & w) >> 2
        i *= 0xC860A3DF
        i &= w
        if i < l:
            break

    return (i + p) % l
