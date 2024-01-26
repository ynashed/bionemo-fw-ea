# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Sequence

import click

from bionemo.triton.serve_bionemo_model import main
from bionemo.triton.types_constants import BIONEMO_MODEL


__all_: Sequence[str] = ()


@click.command()
@click.option('--config-path', required=True, help="Path to Hydra config directory where configuration date lives.")
@click.option(
    '--config-name',
    default='infer.yaml',
    show_default=True,
    required=True,
    help="Name of YAML config file in --config-path to load from.",
)
@click.option(
    '--nav', is_flag=True, help="If present, load runtime optimized with model navigator. Requires export beforehand."
)
def entrypoint(config_path: str, config_name: str, nav: bool) -> None:  # pragma: no cover
    main(config_path=config_path, config_name=config_name, nav=nav, hidden=BIONEMO_MODEL)


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
