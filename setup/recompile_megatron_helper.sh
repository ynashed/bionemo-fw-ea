#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# If there are C-like issues when running on cluster environments, it may be
# necessary to recompile the Megatron helpers, which can be done by running
# this script. This recompilation should also be done immediately before
# training starts on clusters as a best practice.

# Find NeMo installation location and re-combile Megatron helpers
NEMO_PATH=$(python -c 'import nemo; print(nemo.__path__[0])')
cd ${NEMO_PATH}/collections/nlp/data/language_modeling/megatron
make

