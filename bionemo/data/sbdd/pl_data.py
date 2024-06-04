# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data


FOLLOW_BATCH: Sequence[str] = (
    'protein_element',
    'ligand_element',
    'ligand_bond_type',
)


class ProteinLigandData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(
        protein_dict: Optional[Dict[str, Any]] = None, ligand_dict: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        Constructs a ProteinLigandComplex object from dictionaries containing protein and ligand information.

        Args:
            protein_dict (Optional[Dict[str, Any]]): A dictionary containing protein information.
                Expected keys and data types:
                    - 'element': np.ndarray(dtype=np.int_) - Atomic numbers of protein atoms.
                    - 'molecule_name': str - Name of the protein molecule.
                    - 'pos': np.ndarray(dtype=np.float32) - Positions of protein atoms.
                    - 'is_backbone': np.ndarray(dtype=np.bool_) - Boolean array indicating whether each atom is part of the backbone.
                    - 'atom_name': list - Names of protein atoms.
                    - 'atom_to_aa_type': np.ndarray(dtype=np.int_) - Mapping of atom indices to amino acid types.
            ligand_dict (Optional[Dict[str, Any]]): A dictionary containing ligand information.
                Expected keys and data types:
                    - 'smiles': str - SMILES representation of the ligand molecule.
                    - 'element': np.ndarray(dtype=int) - Atomic numbers of ligand atoms.
                    - 'pos': np.ndarray(dtype=np.float32) - Positions of ligand atoms.
                    - 'bond_index': np.ndarray(dtype=np.int_) - Indices of bonded atoms.
                    - 'bond_type': np.ndarray(dtype=np.int_) - Types of bonds.
                    - 'center_of_mass': np.ndarray(dtype=np.float32) - Center of mass of the ligand.
                    - 'atom_feature': np.ndarray(dtype=np.compat.long) - Feature matrix of ligand atoms.
                    - 'hybridization': list - Hybridization states of ligand atoms.
            **kwargs: Additional keyword arguments to be passed.

        Returns:
            ProteinLigandComplex: A ProteinLigandComplex object constructed from the provided dictionaries.
        """
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance[f'protein_{key}'] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance[f'ligand_{key}'] = item

        instance['ligand_nbh_list'] = {
            i.item(): [
                j.item()
                for k, j in enumerate(instance.ligand_bond_index[1])
                if instance.ligand_bond_index[0, k].item() == i
            ]
            for i in instance.ligand_bond_index[0]
        }
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value)


def torchify_dict(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output
