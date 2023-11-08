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


from pathlib import PosixPath

import numpy as np
from pytorch_lightning.callbacks import Callback

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.data.protein.openfold.protein import Protein


class PredictionPDBWriter(Callback):
    """Takes inference output, converts it to Protein instance and writes
    to a PDB file
    """

    def __init__(self, result_path: str):
        self.result_path = PosixPath(result_path)
        self.result_path.mkdir(exist_ok=True, parents=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            batch = [batch]
            outputs = [outputs]
        for input_dict, output_dict in zip(batch, outputs):
            aatype = input_dict["aatype"][0, :, 0].cpu().numpy()
            final_atom_positions = output_dict["final_atom_positions"][0].cpu().numpy()
            final_atom_mask = output_dict["final_atom_mask"][0].cpu().numpy()
            residue_index = input_dict["residue_index"][0, :, 0].cpu().numpy()
            b_factors = np.repeat(
                output_dict["plddt"][0].cpu().numpy()[:, None],
                repeats=rc.ATOM_TYPE_NUM,
                axis=-1,
            )
            unrelaxed_protein = Protein.from_prediction(
                aatype=aatype,
                final_atom_positions=final_atom_positions,
                final_atom_mask=final_atom_mask,
                residue_index=residue_index,
                b_factors=b_factors,
            )
            unrelaxed_pdb_string = unrelaxed_protein.to_pdb_string()
            try:
                name = input_dict['seq_name'][0]
            except KeyError:
                name = input_dict['seq_index'][0]
            unrelaxed_pdb_filepath = self.result_path / f"{name}.pdb"
            with open(unrelaxed_pdb_filepath, "w") as f:
                f.write(unrelaxed_pdb_string)
