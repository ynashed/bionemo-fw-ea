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


import argparse
import random
import string

from scdl.io.SCCollection import SingleCellCollection


def generate_random_string(length):
    return "".join(random.choice(string.ascii_uppercase) for _ in range(length))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers", type=int, default=4, help="The number of AnnData loaders to run in parallel [4]."
    )
    parser.add_argument(
        "--use-mp",
        action="store_true",
        default=False,
        help="Use a subprocess for each worker rather than a lightweight OS thread [False].",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="A path containing AnnData files. Note: These will all be concatenated.",
    )
    parser.add_argument(
        "--save-path", required=True, type=str, help="An output path where an SCDataset will be stored."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tmp_dir = generate_random_string(10)
    coll = SingleCellCollection(tmp_dir)
    coll.load_h5ad_multi(args.data_path, max_workers=args.num_workers, use_processes=args.use_mp)
    coll.flatten(args.save_path, check_column_compatibility=True, check_column_order=True, destroy_on_copy=True)
