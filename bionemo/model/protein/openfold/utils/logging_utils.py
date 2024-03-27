# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nemo.utils import logging


DEFAULT_TAG = "******** bnmo_debug ********"


def log_with_nemo_at_debug(input_string, tag=DEFAULT_TAG):
    logging.debug(prefix_string_with_tag(input_string, tag))


def prefix_string_with_tag(input_string, tag=DEFAULT_TAG):
    return f"{tag}, {input_string}"
