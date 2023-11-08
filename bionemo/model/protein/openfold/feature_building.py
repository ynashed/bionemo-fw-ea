# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# Copyright 2023 NVIDIA CORPORATION
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

from typing import Dict

import torch
import torch.nn.functional as F

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.model.protein.openfold.utils.rigid_utils import Rigid


def _pseudo_beta_fn(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
) -> torch.Tensor:
    is_gly = torch.eq(aatype, rc.RESTYPE_ORDER["G"])
    ca_idx = rc.ATOM_ORDER["CA"]
    cb_idx = rc.ATOM_ORDER["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly.unsqueeze(-1), [1] * is_gly.ndim + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )
    return pseudo_beta


def _build_extra_msa_feat(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    msa_1hot = F.one_hot(input=feats["extra_msa"], num_classes=23)
    msa_feat = [
        msa_1hot,
        feats["extra_has_deletion"].unsqueeze(-1),
        feats["extra_deletion_value"].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


def _build_template_pair_feat(
    feats: Dict[str, torch.Tensor],
    min_bin: int,
    max_bin: int,
    num_bins: int,
    use_unit_vector: bool,
    inf: float,
    eps: float,
) -> torch.Tensor:
    template_mask = feats["template_pseudo_beta_mask"]
    template_mask_2d = template_mask.unsqueeze(-1) * template_mask.unsqueeze(-2)

    # Compute distogram (this seems to differ slightly from Alg. 5)
    tpb = feats["template_pseudo_beta"]
    dgram = torch.sum(
        input=(tpb.unsqueeze(-2) - tpb.unsqueeze(-3)) ** 2,
        dim=-1,
        keepdim=True,
    )
    lower = (
        torch.linspace(
            start=min_bin,
            end=max_bin,
            steps=num_bins,
            device=tpb.device,
        )
        ** 2
    )
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).to(dtype=dgram.dtype)

    to_concat = [dgram, template_mask_2d.unsqueeze(-1)]

    aatype_one_hot = F.one_hot(
        input=feats["template_aatype"],
        num_classes=(rc.RESTYPE_NUM + 2),
    )

    N_res = feats["template_aatype"].shape[-1]

    to_concat.append(aatype_one_hot.unsqueeze(-3).expand(*aatype_one_hot.shape[:-2], N_res, -1, -1))
    to_concat.append(aatype_one_hot.unsqueeze(-2).expand(*aatype_one_hot.shape[:-2], -1, N_res, -1))

    n, ca, c = [rc.ATOM_ORDER[a] for a in ["N", "CA", "C"]]
    rigids = Rigid.make_transform_from_reference(
        n_xyz=feats["template_all_atom_positions"][..., n, :],
        ca_xyz=feats["template_all_atom_positions"][..., ca, :],
        c_xyz=feats["template_all_atom_positions"][..., c, :],
        eps=eps,
    )
    points = rigids.get_trans().unsqueeze(-3)
    rigid_vec = rigids.unsqueeze(-1).invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))

    t_aa_masks = feats["template_all_atom_mask"]
    template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    template_mask_2d = template_mask.unsqueeze(-1) * template_mask.unsqueeze(-2)

    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar.unsqueeze(-1)

    if not use_unit_vector:
        unit_vector = unit_vector * 0.0

    to_concat.extend(torch.unbind(unit_vector.unsqueeze(-2), dim=-1))
    to_concat.append(template_mask_2d.unsqueeze(-1))

    t = torch.cat(to_concat, dim=-1)
    t = t * template_mask_2d.unsqueeze(-1)

    return t


def _build_template_angle_feat(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    template_aatype = feats["template_aatype"]
    torsion_angles_sin_cos = feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = feats["template_alt_torsion_angles_sin_cos"]
    torsion_angles_mask = feats["template_torsion_angles_mask"]
    template_angle_feat = torch.cat(
        [
            F.one_hot(input=template_aatype, num_classes=22),
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14),
            torsion_angles_mask,
        ],
        dim=-1,
    )
    return template_angle_feat


def _atom14_to_atom37(
    atom14_positions: torch.Tensor,
    residx_atom37_to_atom14: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
) -> torch.Tensor:
    # atom14_positions: [batch, N_res, 14, 3]
    # residx_atom37_to_atom14: [batch, N_res, 37]
    # atom37_atom_exists: [batch, N_res, 37]

    indices = residx_atom37_to_atom14.unsqueeze(-1)
    # indices: [batch, N_res, 37, 1]
    indices = indices.expand(-1, -1, -1, 3)
    # indices: [batch, N_res, 37, 3]

    atom37_positions = torch.gather(atom14_positions, 2, indices)
    # atom37_positions: [batch, N_res, 37, 3]

    atom37_mask = atom37_atom_exists.unsqueeze(-1)
    # atom37_mask: [batch, N_res, 37, 1]

    atom37_positions = atom37_positions * atom37_mask
    # atom37_positions: [batch, N_res, 37, 3]

    return atom37_positions
