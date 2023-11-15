# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from typing import Tuple

import dgl
import numpy as np
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.core import LightningModule

from bionemo.data.equidock.protein_utils import (
    get_residues,
    preprocess_unbound_bound,
    protein_to_graph_unbound_bound,
)
from bionemo.model.protein.equidock.equidock_model import EquiDock
from bionemo.model.utils import _reconfigure_microbatch_calculator, parallel_state, setup_inference_trainer


def restore_model(restore_path, trainer=None, cfg=None, model_cls=None, adjust_config=True):
    """Restore model from checkpoint"""
    logging.info(f"Restoring model from {restore_path}")

    # infer model_cls from cfg if missing
    if model_cls is None:
        logging.info(f"Loading model class: {cfg.model.target}")
        model_cls = import_class_by_path(cfg.model.target)

    # merge the config from restore_path with the provided config
    restore_cfg = model_cls.restore_from(
        restore_path=restore_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        return_config=True,
    )
    # build trainer if not provided
    if trainer is None:
        trainer = setup_inference_trainer(cfg=cfg, adjust_config=adjust_config)

    # overwrite model config from checkpoint
    with open_dict(cfg):
        cfg.model = OmegaConf.merge(cfg.model, restore_cfg)

    # enforce trainer precition
    with open_dict(cfg):
        cfg.trainer.precision = trainer.precision

    model = model_cls.restore_from(
        restore_path=restore_path,
        trainer=trainer,
        override_config_path=cfg,
        save_restore_connector=NLPSaveRestoreConnector(),
    )

    return model


class EquiDockInference(LightningModule):
    """
    Base class for inference.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = self.load_model(cfg)
        self._trainer = self.model.trainer

    def load_model(self, cfg):
        """Load saved model checkpoint
        Params:
            checkpoint_path: path to nemo checkpoint
        Returns:
            Loaded model
        """
        # load model class from config which is required to load the .nemo file
        model = restore_model(restore_path=cfg.model.restore_from_path, cfg=cfg, model_cls=EquiDock)

        # move self to same device as loaded model
        self.to(model.device)

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():
            logging.info("DDP is not initialized. Initializing...")

            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()

        # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
        _reconfigure_microbatch_calculator(
            rank=0,  # This doesn't matter since it is only used for logging
            rampup_batch_size=None,
            global_batch_size=1,
            # Make sure that there is no "grad acc" while decoding.
            micro_batch_size=1,
            # We check above to make sure that dataparallel size is always 1 at inference.
            data_parallel_size=1,
        )
        model.freeze()
        self.model = model
        return model

    def create_ligand_receptor_graphs_arrays(
        self, ligand_filename: str, receptor_filename: str
    ) -> Tuple[dgl.graph, dgl.graph, np.ndarray, np.ndarray]:
        """Creates ligand and receptor graphs and arrays (coordinates) from ligand and receptor files"""

        # Preprocess ligand and receptor
        (
            unbound_predic_ligand,
            unbound_predic_receptor,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
        ) = preprocess_unbound_bound(
            get_residues(ligand_filename),
            get_residues(receptor_filename),
            graph_nodes=self.cfg.model.graph_nodes,
            pos_cutoff=self.cfg.data.pocket_cutoff,
            inference=True,
        )

        # Make graphs
        ligand_graph, receptor_graph = protein_to_graph_unbound_bound(
            unbound_predic_ligand,
            unbound_predic_receptor,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
            graph_nodes=self.cfg.model.graph_nodes,
            cutoff=self.cfg.data.graph_cutoff,
            max_neighbor=self.cfg.data.graph_max_neighbor,
            one_hot=False,
            residue_loc_is_alphaC=self.cfg.model.graph_residue_loc_is_alphaC,
        )

        if self.cfg.model.input_edge_feats_dim < 0:
            self.cfg.model.input_edge_feats_dim = ligand_graph.edata['he'].shape[1]

        ligand_graph.ndata['new_x'] = ligand_graph.ndata['x']

        assert (
            np.linalg.norm(bound_ligand_repres_nodes_loc_clean_array - ligand_graph.ndata['x'].detach().cpu().numpy())
            < 1e-1
        )

        return (
            ligand_graph,
            receptor_graph,
            bound_ligand_repres_nodes_loc_clean_array,
            bound_receptor_repres_nodes_loc_clean_array,
        )

    def forward(self, batch):
        """Forward pass of the model"""
        return self.model.net(batch)
