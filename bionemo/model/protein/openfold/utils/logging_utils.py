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
import sys

from nemo.core import ModelPT
from nemo.utils import logging


DEFAULT_TAG = "******** bnmo_debug ********"


def log_with_nemo_at_debug(input_string: str, model_pt: ModelPT = None, tag: str = DEFAULT_TAG) -> str:
    frame_of_caller = sys._getframe(1)
    output_string = f"""
        {input_string}
        frame_of_caller.f_code.co_name={frame_of_caller.f_code.co_name}
        frame_of_caller.f_code.co_filename={frame_of_caller.f_code.co_filename}
        frame_of_caller.f_lineno={frame_of_caller.f_lineno}
    """

    if model_pt:
        # Neither model_pt or model_pt.trainer contains step
        output_string = f"""
        {output_string}
        model_pt.trainer.global_step={model_pt.trainer.global_step}
        model_pt.trainer.global_rank={model_pt.trainer.global_rank}
        """

    logging.debug(prefix_string_with_tag(output_string, tag))


def prefix_string_with_tag(input_string: str, tag: str = DEFAULT_TAG) -> str:
    return f"{tag}, {input_string}"


def environ_as_multiline_str() -> str:
    return "\n".join([str(pair[0]) + ":" + str(pair[1]) for pair in os.environ.items()])
