# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, Literal, Optional, Tuple

import torch
from torch.utils.data import Subset

from bionemo.data.sbdd.pl_pair_dataset import PocketLigandPairDataset


def get_dataset(
    name: Literal['pl'], root: str, split: Optional[str], *args, **kwargs
) -> Tuple[PocketLigandPairDataset, Optional[Dict[str, Subset]]]:
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' != None:
        split = torch.load(split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset, None
