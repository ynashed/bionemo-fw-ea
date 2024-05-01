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


# Build wheel files for PyG packages:
# torch geometric, torch cluster, torch sparse, torch scatter and torch spline conv
# and upload to gitlab package registry, to reduce the time cost when building container
# Usage: bash build_wheels.sh <base container> <path to save wheel files>

BASE_IMAGE=${1:-nvcr.io/nvidia/nemo:23.10}
WHEEL_FILE_PATH=${2:-$(pwd)/build-dir}

set -euo pipefail

packages=" \
 git+https://github.com/rusty1s/pytorch_cluster.git@1.6.3 \
 torch-sparse==0.6.18 \
 git+https://github.com/pyg-team/pytorch_geometric.git@2.5.0 \
 git+https://github.com/rusty1s/pytorch_scatter.git@2.1.2 \
 torch-spline-conv==1.2.2 \
"

CMD="\
  cd /build-dir;
  for package in ${packages}; do \
  echo Building \${package}...; \
  pip wheel --no-deps \${package}; \
  done
"

mkdir -p $WHEEL_FILE_PATH

docker run -v ${WHEEL_FILE_PATH}:/build-dir ${BASE_IMAGE} bash -c "${CMD}"

echo "All wheels ready in ${WHEEL_FILE_PATH} !"
echo "You can now publish them with:"
echo "TWINE_PASSWORD=<gitlab token> TWINE_USERNAME=<gitlab username> \
python -m twine upload --repository-url https://gitlab-master.nvidia.com/api/v4/projects/65301/packages/pypi \
<path to .whl file>"