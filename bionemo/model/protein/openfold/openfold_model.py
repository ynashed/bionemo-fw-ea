# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
from typing import Dict, Tuple, Union

import omegaconf
import torch
from nemo.core import ModelPT
from pytorch_lightning import Trainer

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.data.protein.openfold.datahub import get_training_dataloader, get_validation_dataloader
from bionemo.model.protein.openfold.auxiliary_heads import AuxiliaryHeads
from bionemo.model.protein.openfold.evoformer_stack import EvoformerStack
from bionemo.model.protein.openfold.extra_msa_embedder import ExtraMSAEmbedder
from bionemo.model.protein.openfold.extra_msa_stack import ExtraMSAStack
from bionemo.model.protein.openfold.feature_building import (
    _apply_template_mask,
    _atom14_to_atom37,
    _build_extra_msa_feat,
    _build_template_angle_feat,
    _pseudo_beta,
)
from bionemo.model.protein.openfold.input_embedder import InputEmbedder
from bionemo.model.protein.openfold.loss import AlphaFoldLoss
from bionemo.model.protein.openfold.recycling_embedder import RecyclingEmbedder
from bionemo.model.protein.openfold.structure_module import StructureModule
from bionemo.model.protein.openfold.template_angle_embedder import TemplateAngleEmbedder
from bionemo.model.protein.openfold.template_pair_embedder import TemplatePairEmbedder
from bionemo.model.protein.openfold.template_pair_stack import TemplatePairStack
from bionemo.model.protein.openfold.template_pointwise_attention import TemplatePointwiseAttention
from bionemo.model.protein.openfold.utils.logging_utils import log_with_nemo_at_debug
from bionemo.model.protein.openfold.utils.torch_utils import map_tensor_tree
from bionemo.model.protein.openfold.validation_metrics import compute_validation_metrics


VAL_EXAMPLE_ID = "val_example_id"
VAL_EXAMPLE_ID_TENSOR_POSN_TO_INSERT = "val_example_id_tensor_posn_to_insert"


class AlphaFold(ModelPT):
    def __init__(self, cfg: Union[omegaconf.DictConfig, Dict], trainer: Trainer, *args, **kwarg) -> None:
        r"""OpenFold faithful reproduction of AlphaFold2

        AlphaFold2: https://www.nature.com/articles/s41586-021-03819-2
        OpenFold: https://www.biorxiv.org/content/10.1101/2022.11.20.517210v1.full.pdf

        A - https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/core/module.py
        B -  https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/common.py#L672

                    A                                     B
        pytorch_lightning.LightningModule     nemo.core.classes.common.Model
                                      \        /
                                  nemo.core.ModelPT
                                        |
                                    AlphaFold

        Notes:
            1. self.global_step: int, is inherited from ModelPT
            2. logging module:
                https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/nemo_logging.py
                switch to debug verbosity with
                https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/nemo_logging.py#L121C32-L121C56
                or setLevel():
                https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/nemo_logging.py#L225

        Args:
            cfg (Union[omegaconf.DictConfig, Dict]): model config containing architectural, loss,
            and data parameters.

            Please refer to the following configs inside bionemo.model.protein.openfold.conf:
                1. base_config - for all architecture and loss parameters.
                2. openfold_initial_training - for overrides required to run initial-training
                stage as described in AlphaFold2 paper
                3. openfold_finetuning - for overrides required to run fine-tuning
                stage as described in AlphaFold2 paper

            trainer (Trainer): PyTorch Lightning Trainer
        """
        self.cfg = cfg
        super(AlphaFold, self).__init__(cfg, trainer, *args, **kwarg)
        self.input_embedder = InputEmbedder(
            **(cfg.input_embedder_config),
        )
        self.recycling_embedder = RecyclingEmbedder(
            **(cfg.recycling_embedder_config),
        )
        if cfg.templates_enabled:
            self.template_angle_embedder = TemplateAngleEmbedder(
                **(cfg.template_angle_embedder_config),
            )
            self.template_pair_embedder = TemplatePairEmbedder(
                **(cfg.template_pair_embedder_config),
            )
            self.template_pair_stack = TemplatePairStack(
                **(cfg.template_pair_stack_config),
            )
            self.template_pointwise_attention = TemplatePointwiseAttention(
                **(cfg.template_pointwise_attention_config),
            )
        self.extra_msa_embedder = ExtraMSAEmbedder(
            **(cfg.extra_msa_embedder_config),
        )
        self.extra_msa_stack = ExtraMSAStack(
            **(cfg.extra_msa_stack_config),
        )
        self.evoformer_stack = EvoformerStack(
            **(cfg.evoformer_stack_config),
        )
        self.structure_module = StructureModule(
            **(cfg.structure_module_config),
        )
        self.auxiliary_heads = AuxiliaryHeads(cfg.auxiliary_heads_config)
        self.loss = AlphaFoldLoss(config=cfg.loss_config)
        self.multisessionstep_from_checkpoint_filename = None
        self.val_metrics_in_model_as_tensors = {}

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Initialize previous recycling embeddings:
        prevs = self._initialize_prevs(batch)

        # Forward iterations with autograd disabled:
        num_recycling_iters = batch["aatype"].shape[-1] - 1
        for j in range(num_recycling_iters):
            feats = map_tensor_tree(fn=lambda t: t[..., j].contiguous(), tree=batch)
            with torch.no_grad():
                outputs, prevs = self._forward_iteration(
                    feats=feats,
                    prevs=prevs,
                    gradient_checkpointing=False,
                )
                del outputs

        # https://github.com/pytorch/pytorch/issues/65766
        if torch.is_autocast_enabled():
            torch.clear_autocast_cache()

        # Final iteration with autograd enabled:
        feats = map_tensor_tree(fn=lambda t: t[..., -1].contiguous(), tree=batch)
        outputs, prevs = self._forward_iteration(
            feats=feats,
            prevs=prevs,
            gradient_checkpointing=self.training,
        )
        del prevs

        # Run auxiliary heads:

        outputs["msa"] = outputs["msa"].to(dtype=torch.float32)
        outputs["pair"] = outputs["pair"].to(dtype=torch.float32)
        outputs["single"] = outputs["single"].to(dtype=torch.float32)

        aux_outputs = self.auxiliary_heads(outputs)
        outputs.update(aux_outputs)

        return outputs

    def _forward_iteration(
        self,
        feats: Dict[str, torch.Tensor],
        prevs: Dict[str, torch.Tensor],
        gradient_checkpointing: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        outputs = {}

        N_clust = feats["msa_feat"].shape[1]

        seq_mask = feats["seq_mask"]
        # seq_mask: [batch, N_res]

        pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)  # outer product
        # pair_mask: [batch, N_res, N_res]

        msa_mask = feats["msa_mask"]
        # msa_mask: [batch, N_clust, N_res]

        # Initialize MSA and pair representations:
        m, z = self.input_embedder(
            target_feat=feats["target_feat"],
            residue_index=feats["residue_index"],
            msa_feat=feats["msa_feat"],
        )
        # m: [batch, N_clust, N_res, c_m]
        # z: [batch, N_res, N_res, c_z]

        # Extract recycled representations:
        m0_prev = prevs.pop("m0_prev", None)
        z_prev = prevs.pop("z_prev", None)
        x_prev = prevs.pop("x_prev", None)

        x_prev = _pseudo_beta(
            aatype=feats["aatype"],
            all_atom_positions=x_prev,
            dtype=z.dtype,
        )

        m, z = self.recycling_embedder(
            m=m,
            z=z,
            m0_prev=m0_prev,
            z_prev=z_prev,
            x_prev=x_prev,
        )
        # m0_prev_emb: [batch, N_res, c_m]
        # z_prev_emb: [batch, N_res, N_res, c_z]

        del m0_prev, z_prev, x_prev

        # Embed templates and merge with MSA/pair representation:
        if self.cfg.templates_enabled:
            template_feats = {k: t for k, t in feats.items() if k.startswith("template_")}
            template_embeds = self._embed_templates(
                feats=template_feats,
                z=z,
                pair_mask=pair_mask,
                gradient_checkpointing=gradient_checkpointing,
            )

            z = z + template_embeds["template_pair_embedding"]
            # z: [batch, N_res, N_res, c_z]

            if self.cfg.embed_template_torsion_angles:
                m = torch.cat([m, template_embeds["template_angle_embedding"]], dim=1)
                # m: [batch, N_seq, N_res, c_m]

                msa_mask = torch.cat(
                    [
                        feats["msa_mask"],
                        feats["template_torsion_angles_mask"][..., 2],
                    ],
                    dim=-2,
                )
                # msa_mask: [batch, N_seq, N_res]

            del template_feats, template_embeds

        # N_seq = m.shape[1]

        # Embed extra MSA features and merge with pairwise embeddings:
        # N_extra_seq = feats["extra_msa"].shape[1]
        a = self.extra_msa_embedder(_build_extra_msa_feat(feats))
        # a: [batch, N_extra_seq, N_res, c_e]
        z = self.extra_msa_stack(
            m=a,
            z=z,
            msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=m.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # z: [batch, N_res, N_res, c_z]
        del a

        # Evoformer forward pass:
        m, z, s = self.evoformer_stack(
            m=m,
            z=z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # m: [batch, N_seq, N_res, c_m]
        # z: [batch, N_res, N_res, c_z]
        # s: [batch, N_res, c_s]
        outputs["msa"] = m[:, :N_clust]
        outputs["pair"] = z
        outputs["single"] = s

        # Predict 3D structure:
        sm_outputs = self.structure_module(
            s=outputs["single"].to(dtype=torch.float32),
            z=outputs["pair"].to(dtype=torch.float32),
            mask=feats["seq_mask"].to(dtype=s.dtype),
            aatype=feats["aatype"],
        )
        outputs.update(sm_outputs)
        outputs["final_atom_positions"] = _atom14_to_atom37(
            atom14_positions=outputs["sm_positions"][:, -1],
            residx_atom37_to_atom14=feats["residx_atom37_to_atom14"],
            atom37_atom_exists=feats["atom37_atom_exists"],
        )

        # TODO: [optim-hub] why do we need to align dtypes?
        outputs["final_atom_mask"] = feats["atom37_atom_exists"].to(dtype=outputs["final_atom_positions"].dtype)
        outputs["final_affine_tensor"] = outputs["sm_frames"][:, -1]

        # Save embeddings for next recycling iteration:
        prevs = {}
        prevs["m0_prev"] = m[:, 0]
        prevs["z_prev"] = outputs["pair"]
        prevs["x_prev"] = outputs["final_atom_positions"]

        return outputs, prevs

    def _embed_templates(
        self,
        feats: Dict[str, torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        gradient_checkpointing: bool,
    ) -> Dict[str, torch.Tensor]:
        # Embed the templates one at a time:
        pair_embeds = []
        N_templ = feats["template_aatype"].shape[1]
        for i in range(N_templ):
            single_template_feats = map_tensor_tree(fn=lambda t: t[:, i], tree=feats)
            t = self.template_pair_embedder.build_template_pair_feat(
                feats=single_template_feats,
                min_bin=self.cfg.template_pair_feat_distogram_min_bin,
                max_bin=self.cfg.template_pair_feat_distogram_max_bin,
                num_bins=self.cfg.template_pair_feat_distogram_num_bins,
                use_unit_vector=self.cfg.template_pair_feat_use_unit_vector,
                inf=self.cfg.template_pair_feat_inf,
                eps=self.cfg.template_pair_feat_eps,
                dtype=z.dtype,
            )
            t = self.template_pair_embedder(t)
            # t: [batch, N_res, N_res, c_t]
            pair_embeds.append(t)
            del t

        t = torch.stack(pair_embeds, dim=1)
        # t: [batch, N_templ, N_res, N_res, c_t]
        del pair_embeds

        t = self.template_pair_stack(
            t=t,
            mask=pair_mask.to(dtype=z.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # t: [batch, N_templ, N_res, N_res, c_t]

        t = self.template_pointwise_attention(
            t=t,
            z=z,
            template_mask=feats["template_mask"].to(dtype=z.dtype),
        )
        # t: [batch, N_res, N_res, c_z]

        t = _apply_template_mask(t=t, template_mask=feats["template_mask"])
        # t: [batch, N_res, N_res, c_z]

        template_embeds = {}
        template_embeds["template_pair_embedding"] = t

        if self.cfg.embed_template_torsion_angles:
            template_angle_feat = _build_template_angle_feat(feats)
            a = self.template_angle_embedder(template_angle_feat)
            # a: [batch, N_templ, N_res, c_m]
            template_embeds["template_angle_embedding"] = a

        return template_embeds

    def _initialize_prevs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        prevs = {}
        batch_size = batch["aatype"].shape[0]
        N_res = batch["aatype"].shape[1]
        c_m = self.input_embedder.c_m
        c_z = self.input_embedder.c_z
        device = batch["msa_feat"].device
        dtype = batch["msa_feat"].dtype
        prevs["m0_prev"] = torch.zeros(
            size=[batch_size, N_res, c_m],
            device=device,
            dtype=dtype,
        )
        prevs["z_prev"] = torch.zeros(
            size=[batch_size, N_res, N_res, c_z],
            device=device,
            dtype=dtype,
        )
        prevs["x_prev"] = torch.zeros(
            size=[batch_size, N_res, rc.ATOM_TYPE_NUM, 3],
            device=device,
            dtype=torch.float32,
        )
        return prevs

    def get_checkpoint_step(self) -> int:
        """In DDP, checkpoints are loaded after data setup. This means information about
        global step is unavailable while recreating data loaders. Here, we decode step
        from checkpoint name and deduct 1 since iterations are calculated starting from 1
        and data loaders index from 0
        Returns:
            int: iteration number
        """
        log_with_nemo_at_debug(
            f"""
            AlphaFold.get_checkpoint_step(),
            self.trainer.ckpt_path={self.trainer.ckpt_path}
            """
        )
        if self.trainer.ckpt_path:
            self.multisessionstep_from_checkpoint_filename = int(
                float(re.search(r".*multisessionstep=(\d+.\d+)--.*", self.trainer.ckpt_path).group(1))
            )
            step_from_checkpoint_filename = int(re.search(r".*--step=(\d+)--.*", self.trainer.ckpt_path).group(1))
            log_with_nemo_at_debug(
                f"""
                AlphaFold.get_checkpoint_step(),
                multisessionstep_from_checkpoint_filename={self.multisessionstep_from_checkpoint_filename}
                step_from_checkpoint_filename={step_from_checkpoint_filename}
                """
            )
            return self.multisessionstep_from_checkpoint_filename - 1

        else:
            return 0

    def get_dist_setup(self):
        return omegaconf.OmegaConf.create(
            {'rank': self.global_rank, 'world_size': self.world_size, 'iteration': self.get_checkpoint_step()}
        )

    def setup_training_data(self, train_data_cfg: Union[omegaconf.DictConfig, Dict]):
        dist_setup = self.get_dist_setup()
        self._train_dl = get_training_dataloader(
            model_cfg=self.cfg, train_session_cfg=dist_setup, ds_cfg=train_data_cfg
        )

    def setup_validation_data(self, val_data_cfg: Union[omegaconf.DictConfig, Dict]):
        dist_setup = self.get_dist_setup()
        self._validation_dl = get_validation_dataloader(model_cfg=self.cfg, dist_cfg=dist_setup, ds_cfg=val_data_cfg)
        log_with_nemo_at_debug(
            f"""
            AlphaFold.setup_validation_data(),
            self._validation_dl.sampler._dataset_length={self._validation_dl.sampler._dataset_length}
            self._validation_dl.sampler._epoch_length={self._validation_dl.sampler._epoch_length}
            """
        )

    def training_step(self, train_batch, train_batch_idx, train_dataset_idx=None):
        """For this model, this method is always called with train_dataset_idx=None

        This is called for global_step=0
        """
        torch.manual_seed(self.cfg.seed + self.global_step)
        outputs = self(train_batch)
        step_loss, step_losses = self.loss(
            outputs=outputs,
            batch=map_tensor_tree(fn=lambda t: t[..., -1], tree=train_batch),
        )
        self.log_dict(step_losses, prog_bar=True)
        return step_loss

    def validation_step(self, val_batch, val_batch_idx, val_dataset_idx=None):
        """For this model, this method is always called with val_dataset_idx=None.

        Args:
            val_batch (dict):
            val_batch_idx (int):
            val_dataset_idx: This is

        """
        # (0) log data record data debug level,
        #   - activate with export NEMO_TESTING=True
        log_with_nemo_at_debug(
            f"""
            AlphaFold.validation_step(),
            self.trainer.global_step={self.trainer.global_step}
            val_batch_idx={val_batch_idx}
            val_dataset_idx={val_dataset_idx}
            val_batch["id"]={val_batch.get("id", None)}
            """
        )
        # (1)
        self._update_and_log_multisessionstep_this_validation_step()

        # (2) val metrics, this global_rank, this step
        val_metrics_this_step_this_rank = self._compute_validation_metrics_this_validation_step(
            val_batch, val_batch_idx
        )

        # (2) custom val metrics data structures
        self._update_val_metrics_this_validation_step(val_batch, val_metrics_this_step_this_rank)

    def on_validation_epoch_end(self):
        super(AlphaFold, self).on_validation_epoch_end()
        self._manage_val_metrics_on_validation_epoch_end()

    def _compute_validation_metrics_this_validation_step(self, val_batch, val_batch_idx):
        val_outputs = self(val_batch)
        val_batch = map_tensor_tree(lambda t: t[..., -1], val_batch)
        val_metrics_this_step_this_rank = compute_validation_metrics(
            predicted_atom_positions=val_outputs["final_atom_positions"],
            target_atom_positions=val_batch["all_atom_positions"],
            atom_mask=val_batch["all_atom_mask"],
            metrics_names=set(self.cfg.metrics),
        )
        for metric_name, metric_value in val_metrics_this_step_this_rank.items():
            log_with_nemo_at_debug(
                f"""
                AlphaFold.validation_step(),
                self.trainer.global_step={self.trainer.global_step}
                val_batch_idx={val_batch_idx}
                global_rank={self.global_rank}
                world_size={self.world_size}
                {metric_name}={metric_value}
                """
            )
        return val_metrics_this_step_this_rank

    def _update_and_log_multisessionstep_this_validation_step(self):
        # (1) progress counters
        multisessionstep = self.global_step
        if self.multisessionstep_from_checkpoint_filename is not None:
            multisessionstep = int(self.multisessionstep_from_checkpoint_filename) + self.global_step

        # (2) logging
        #   - We tried converting multisessionstep to a tensor,
        #   with type torch.int32, before passing to self.log(...),
        #   but the decimal place remains in the checkpoint filenames.
        #   - Will have to overwrite this method to get integer values in filename ?
        #      https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/callbacks/model_checkpoint.py#L567
        #
        log_with_nemo_at_debug(
            f"""
            AlphaFold.validation_step(),
            multisessionstep={multisessionstep}
            self.global_step={self.global_step}
            self.trainer.global_step={self.trainer.global_step}
            global_rank={self.global_rank}
            world_size={self.world_size}
            """
        )
        # call PTL log method
        self.log('multisessionstep', multisessionstep, batch_size=1)

    def _update_val_metrics_this_validation_step(self, val_batch, val_metrics_this_step_this_rank):
        """Deposits the metric values computed for this global_rank at the
        validation step in a tensor, to support an all-gather operation.

        Every val_check_interval training steps, there is a validation epoch.
        For each validation epoch, there are at most num_val_steps_each_val_epoch_max
        calls to self.validation_step(.) for each value of global_rank, i.e.,
        each task, i.e., each gpu.

        -- implement custom aggregation over global rank vlaues
        -- standard PTL aggregation has pitfall

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
        # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metrics-in-distributed-data-parallel-ddp-mode
        # https://lightning.ai/docs/fabric/stable/advanced/distributed_communication.html#barrier



        Args:
            val_batch:
            val_metrics_this_step_this_rank:

        """
        num_val_steps_each_val_epoch_max = self._validation_dl.sampler._epoch_length
        val_example_id = int(val_batch["id"][0][1]) if "id" in val_batch else -1

        # initialize tensors if first call this epoch
        if VAL_EXAMPLE_ID not in self.val_metrics_in_model_as_tensors:
            self.val_metrics_in_model_as_tensors[VAL_EXAMPLE_ID] = torch.tensor(
                num_val_steps_each_val_epoch_max * [-1]
            )
            self.val_metrics_in_model_as_tensors[VAL_EXAMPLE_ID_TENSOR_POSN_TO_INSERT] = torch.tensor([0])
            for metric_name, _ in val_metrics_this_step_this_rank.items():
                self.val_metrics_in_model_as_tensors[metric_name] = torch.tensor(
                    num_val_steps_each_val_epoch_max * [-1.0]
                )

        # populate the tensors
        self.val_metrics_in_model_as_tensors[VAL_EXAMPLE_ID][
            self.val_metrics_in_model_as_tensors[VAL_EXAMPLE_ID_TENSOR_POSN_TO_INSERT]
        ] = val_example_id

        for metric_name, _ in val_metrics_this_step_this_rank.items():
            self.val_metrics_in_model_as_tensors[metric_name][
                self.val_metrics_in_model_as_tensors[VAL_EXAMPLE_ID_TENSOR_POSN_TO_INSERT]
            ] = val_metrics_this_step_this_rank[metric_name]

        # increment the position to insert new values
        self.val_metrics_in_model_as_tensors[VAL_EXAMPLE_ID_TENSOR_POSN_TO_INSERT] += 1

    def _manage_val_metrics_on_validation_epoch_end(self):
        """After all validation steps this epoch, aggregate the metric in a way that
        pays attention to distribution of input examples over the gpu's..

        The all_gather(.) method is discussed in

        https://lightning.ai/docs/fabric/stable/advanced/distributed_communication.html
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        """
        # (0) log some info
        #
        log_with_nemo_at_debug(
            f"""
            AlphaFold.on_validation_epoch_end(),
            self.val_metrics_in_model_as_tensors=\n{self.val_metrics_in_model_as_tensors}
            self.trainer.global_step{self.trainer.global_step}
            self.trainer.current_epoch={self.trainer.current_epoch}
            self.trainer.global_rank={self.trainer.global_rank}
            self.world_size={self.world_size}
            """
        )

        # (1) All-gather
        ptl_gathered_val_metrics_as_tensors = self.all_gather(self.val_metrics_in_model_as_tensors)

        log_with_nemo_at_debug(
            f"""
            AlphaFold.on_validation_epoch_end(),
            ptl_gathered_val_metrics_as_tensors={ptl_gathered_val_metrics_as_tensors}
            self.trainer.global_step{self.trainer.global_step}
            self.trainer.current_epoch={self.trainer.current_epoch}
            self.trainer.global_rank={self.trainer.global_rank}
            self.world_size={self.world_size}
            """
        )

        # (2) aggregate metrics from gathered tensors
        #
        has_val_example = ptl_gathered_val_metrics_as_tensors[VAL_EXAMPLE_ID] != -1
        num_nondistinct_val_example_ids_this_epoch = torch.sum(has_val_example)
        num_distinct_val_example_ids_this_epoch = len(
            {int(x) for x in ptl_gathered_val_metrics_as_tensors[VAL_EXAMPLE_ID][has_val_example].view(-1)}
        )
        metric_means = {
            k: -1.0
            for k, _ in ptl_gathered_val_metrics_as_tensors.items()
            if k not in [VAL_EXAMPLE_ID, VAL_EXAMPLE_ID_TENSOR_POSN_TO_INSERT]
        }
        if num_nondistinct_val_example_ids_this_epoch > 0:
            for metric_name, _ in metric_means.items():
                metric_means[metric_name] = (
                    torch.sum(ptl_gathered_val_metrics_as_tensors[metric_name][has_val_example])
                    / num_nondistinct_val_example_ids_this_epoch
                )

        log_with_nemo_at_debug(
            f"""
            AlphaFold.on_validation_epoch_end(), after compute means
            num_nondistinct_val_example_ids_this_epoch={num_nondistinct_val_example_ids_this_epoch}
            num_distinct_val_example_ids_this_epoch={num_distinct_val_example_ids_this_epoch}
            metric_means={metric_means}
            self.trainer.global_rank={self.trainer.global_rank}
            world_size={self.world_size}
            self.trainer.current_epoch={self.trainer.current_epoch}
            self.trainer.global_step{self.trainer.global_step}
            """
        )

        # (3) PTL logger
        self.log(
            "num_nondistinct_val_example_ids_this_epoch", num_nondistinct_val_example_ids_this_epoch, batch_size=1
        )
        self.log("num_distinct_val_example_ids_this_epoch", num_distinct_val_example_ids_this_epoch, batch_size=1)
        for metric_name, metric_mean_value in metric_means.items():
            self.log(f"{'val'}_{metric_name}", metric_mean_value, batch_size=1)

        # (4) clean up tensors with val metrics
        #    self.val_metrics_in_model_as_tensors = {}
        for _, v in self.val_metrics_in_model_as_tensors.items():
            del v
        self.val_metrics_in_model_as_tensors.clear()

    def list_available_models(self):
        pass
