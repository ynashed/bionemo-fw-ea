# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import logging

from bionemo.model.molecule.infer import MolInference


log = logging.getLogger(__name__)
__all__ = ["MolMIMInference"]


class MolMIMInference(MolInference):
    '''
    Any MolMIM specific inference methods should go here as they arise. Otherwise add them to the parent class if shared.
    '''
