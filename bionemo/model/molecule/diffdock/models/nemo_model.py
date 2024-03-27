# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy

import numpy as np
import torch
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from pytorch_lightning import Trainer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from bionemo.data.diffdock.data_manager import DataManager
from bionemo.data.diffdock.sampler import SizeAwareBatchSampler
from bionemo.model.molecule.diffdock.models.tensor_product_score_model import (
    TensorProductScoreModel,
)
from bionemo.model.molecule.diffdock.models.tensor_product_score_model_all_atom import (
    TensorProductScoreModelAllAtom,
)
from bionemo.model.molecule.diffdock.utils.ddp import get_rank, init_distributed
from bionemo.model.molecule.diffdock.utils.sampling import randomize_position, sampling
from bionemo.model.molecule.diffdock.utils.so3 import score_norm as so3_score_norm
from bionemo.model.molecule.diffdock.utils.utils import estimate_memory_usage


__all__ = ["DiffdockTensorProductScoreModel", "DiffdockTensorProductScoreModelAllAtom"]


class TensorProductScoreModelBase(ModelPT):
    """
    This class contstructs ModelPT on top of the core TensorProductScoreModel (CG or AA).
    """

    def __init__(
        self,
        cfg: OmegaConf,
        net: torch.nn.Module,
        trainer: Trainer = None,
        data_manager: DataManager = None,
    ):
        # TODO does it make sense to allow trainer and/or data_manager to be None?
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices
        else:
            self.world_size = 1
        if self.world_size > 1:
            init_distributed(self.world_size)
        logging.info("Fetching train/test/validation splits of protein-ligand heterographs")
        if data_manager is not None:
            self._train_ds = data_manager.train_ds
            self._validation_ds = data_manager.validation_ds
            self._test_ds = data_manager.test_ds
        else:
            self._train_ds = None
            self._validation_ds = None
            self._test_ds = None
        super().__init__(cfg, trainer)
        self.net = net

        if not self.cfg.confidence_mode and self.cfg.val_denoising_inference_freq is not None:
            self.local_num_denoising_inference_complexes = (
                self.cfg.num_denoising_inference_complexes // self.world_size
                + (get_rank() < self.cfg.num_denoising_inference_complexes % self.world_size)
            )

            self.local_inference_complex_count = 0

    def diffdock_build_dataloader(self, data_config, dataset, use_new_batch_sampler=False, shuffle=False):
        if not use_new_batch_sampler or self.cfg.get('batch_sampler') is None:
            if self.world_size > 1:
                item_sampler = DistributedSampler(dataset, drop_last=False, seed=self.cfg.seed)
                batch_sampler = BatchSampler(item_sampler, batch_size=self.cfg.micro_batch_size, drop_last=False)
                batch_sampler.micro_batch_size = self.cfg.micro_batch_size
                batch_sampler.set_epoch = item_sampler.set_epoch
                return DataLoader(
                    dataset=dataset,
                    num_workers=data_config.num_workers,
                    pin_memory=data_config.pin_memory,
                    batch_sampler=batch_sampler,
                )
            else:
                return DataLoader(
                    dataset=dataset,
                    batch_size=self.cfg.micro_batch_size,
                    num_workers=data_config.num_workers,
                    pin_memory=data_config.pin_memory,
                    drop_last=True,
                    generator=torch.Generator().manual_seed(self.cfg.seed),
                    shuffle=shuffle,
                )
        elif self.cfg.batch_sampler == 'SizeAwareBatchSampler':
            if self.world_size > 1:
                item_sampler = DistributedSampler(dataset, drop_last=False, seed=self.cfg.seed)
            else:
                item_sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(self.cfg.seed))
            estimate_num_cross_edges = self.cfg.estimate_memory_usage.estimate_num_cross_edges

            def num_cross_edge_upper_bound_estimate(n1, n2, n3, n4):
                tmpdict = {'ligand': n1, 'ligand_ligand': n2, 'receptor': n3, 'receptor_receptor': n4}
                num_edges = 0.0
                for term in estimate_num_cross_edges.terms:
                    tmp = term[0]
                    for k in term[1:]:
                        tmp *= tmpdict[k]
                    num_edges += tmp
                num_edges *= estimate_num_cross_edges.scale
                return num_edges

            def estimate_size(g):
                n1, n2, n3, n4 = (
                    g['ligand'].num_nodes,
                    g['ligand', 'ligand'].num_edges,
                    g['receptor'].num_nodes,
                    g['receptor', 'receptor'].num_edges,
                )
                # estimate the upper bound of the number of cross edges
                # the number of cross edges roughly increases w.r.t. the diffusion step t (sampled from uniform(0,1))
                # the empirical formula here is from the polynomial fitting
                # the scaling constant is to help remove the outliers above the upper bound estimation.
                n5 = num_cross_edge_upper_bound_estimate(n1, n2, n3, n4)
                total_memory = estimate_memory_usage(g, n5, self.cfg.estimate_memory_usage, bias=False)
                return total_memory

            if data_config.get('num_batches') is None:
                num_samples = len(item_sampler)
                num_batches = (num_samples // self.cfg.micro_batch_size) + (
                    num_samples % self.cfg.micro_batch_size != 0
                )
            else:
                num_batches = data_config.num_batches
            batch_sampler = SizeAwareBatchSampler(
                item_sampler,
                max_total_size=(
                    self.cfg.max_total_size
                    if self.cfg.get("max_total_size", None) is not None
                    else (0.85 * torch.cuda.get_device_properties('cuda:0').total_memory / 2**20)
                ),
                sizes=estimate_size,
                batch_size=self.cfg.micro_batch_size,
                num_batches=num_batches,
            )

            return DataLoader(
                dataset=dataset,
                num_workers=data_config.num_workers,
                pin_memory=data_config.pin_memory,
                batch_sampler=batch_sampler,
            )
        else:
            raise NotImplementedError(f"batch sampler {self.cfg.batch_sampler}")

    def setup_training_data(self, train_data_config: OmegaConf):
        logging.info(f"Length of train dataset: {len(self._train_ds)}")
        self._train_dl = self.diffdock_build_dataloader(
            train_data_config,
            self._train_ds,
            use_new_batch_sampler=not self.cfg.confidence_mode,
            shuffle=True,
        )

    def setup_validation_data(self, val_data_config: OmegaConf):
        logging.info(f"Length of validation dataset: {len(self._validation_ds)}")
        self._validation_dl = self.diffdock_build_dataloader(
            val_data_config,
            self._validation_ds,
            shuffle=True,
        )

        if not self.cfg.confidence_mode and self.cfg.val_denoising_inference_freq is not None:
            # the inference step needs the ligand pos from self._validation_dl.heterograph_store
            # but we lose the index info, so build a dict to map from name to index
            self._validation_ds_idx_map = {graph.name: idx for idx, graph in enumerate(self._validation_ds)}

    def setup_test_data(self, test_data_config: OmegaConf):
        logging.info(f"Length of test dataset: {len(self._test_ds)}")
        self._test_dl = self.diffdock_build_dataloader(
            test_data_config,
            self._test_ds,
        )

    def training_step(self, train_batch, batch_idx):
        if self.net.confidence_mode:
            pred = self.net.forward(train_batch)
            loss = self.loss_function_confidence(pred, train_batch)
        else:
            tr_pred, rot_pred, tor_pred = self.net.forward(train_batch)
            if tr_pred is None:
                return
            (
                loss,
                tr_loss,
                rot_loss,
                tor_loss,
                tr_base_loss,
                rot_base_loss,
                tor_base_loss,
            ) = self.loss_function(tr_pred, rot_pred, tor_pred, batch=train_batch)

        train_log = {'train_loss': loss.cpu().detach()}
        self.log_dict(
            train_log,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            batch_size=train_batch.num_graphs,
            sync_dist=True,
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.net.confidence_mode:
            pred = self.net.forward(val_batch)
            loss = self.loss_function_confidence(pred, val_batch)
            val_loss = loss.cpu().detach()
        else:
            tr_pred, rot_pred, tor_pred = self.net.forward(val_batch)
            (
                loss,
                tr_loss,
                rot_loss,
                tor_loss,
                tr_base_loss,
                rot_base_loss,
                tor_base_loss,
            ) = self.loss_function(tr_pred, rot_pred, tor_pred, batch=val_batch, apply_mean=False)
            val_loss = loss.mean().cpu().detach()
        if loss.isnan().any():
            logging.warning(f"val_loss is nan... skipping validation batch {batch_idx}")
            return
        val_log = {"val_loss": val_loss}

        self.log_dict(
            val_log,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            batch_size=val_batch.num_graphs,
            sync_dist=True,
        )

        if not self.net.confidence_mode and self.cfg.test_sigma_intervals > 0:
            complex_t_tr, complex_t_rot, complex_t_tor = [
                val_batch.complex_t[noise_type] for noise_type in ["tr", "rot", "tor"]
            ]
            num_intervals = self.cfg.get('num_intervals', 10) or 10
            sigma_index_tr = torch.round(complex_t_tr.cpu() * (num_intervals - 1)).int()

            for k in range(num_intervals):
                indx = sigma_index_tr == k

                if indx.sum() > 0:
                    interval_log = {
                        f"val_loss_interval_{k}": loss.cpu().detach()[indx].mean(),
                    }

                    self.log_dict(
                        interval_log,
                        logger=True,
                        on_epoch=True,
                        batch_size=indx.sum(),
                        sync_dist=True,
                    )

        # Computing RMSDs from the generated ligand positions via reverse diffusion using the traing score model.
        # The steps:
        #     1. Randomize original ligand positions
        #     2. Do reverse diffusion from t: 1->0 with num_inferecen steps
        #     3. Compute the RMSDs from the generated ligand positions
        # DiffDock use valinf_rmsds_lt2 (percent of inference rmsds less than 2.0 Ang on validation dataset)
        # as the metric for the scheduler to select best models as well as reduce the learning rate
        if not self.cfg.confidence_mode and self.cfg.val_denoising_inference_freq is not None:
            if batch_idx == 0:
                self.local_inference_complex_count = 0

            if (
                (self.current_epoch + 1) % self.cfg.val_denoising_inference_freq == 0
                and self.local_inference_complex_count < self.local_num_denoising_inference_complexes
            ):
                batch = Batch.from_data_list(
                    [
                        self._validation_ds.heterograph_store[self._validation_ds_idx_map[name]]
                        for name in val_batch.name[
                            : (self.local_num_denoising_inference_complexes - self.local_inference_complex_count)
                        ]
                    ]
                )

                rmsds = self._denoising_inference_step(batch, batch_idx)

                self.local_inference_complex_count += batch.num_graphs
                inf_metrics = {
                    "valinf_rmsds_lt2": (rmsds < 2).sum() / self.cfg.num_denoising_inference_complexes,
                    "valinf_rmsds_lt5": (rmsds < 5).sum() / self.cfg.num_denoising_inference_complexes,
                }

                self.log_dict(
                    inf_metrics,
                    prog_bar=True,
                    logger=True,
                    on_epoch=True,
                    batch_size=batch.num_graphs,
                    reduce_fx=torch.sum,
                    sync_dist=True,
                )
            elif (
                (self.current_epoch + 1) % self.cfg.val_denoising_inference_freq != 0
                and batch_idx == 0
                and 'valinf_rmsds_lt2' not in self.trainer.callback_metrics
            ):
                # return fake 0 as checkpoint saving is expecting these values
                inf_metrics = {
                    "valinf_rmsds_lt2": 0.0,
                    "valinf_rmsds_lt5": 0.0,
                }
                self.log_dict(
                    inf_metrics,
                    prog_bar=True,
                    logger=True,
                    on_epoch=True,
                    batch_size=val_batch.num_graphs,
                    reduce_fx=torch.sum,
                    sync_dist=True,
                )

    def _denoising_inference_step(self, orig_complex_graphs, batch_idx):
        """
        Computing RMSDs from the generated ligand positions via reverse diffusion using the traing score model.
        The steps:
            1. Randomize original ligand positions
            2. Do reverse diffusion from t: 1->0 with num_inferecen steps
            3. Compute the RMSDs from the generated ligand positions
        """
        t_schedule = np.linspace(1, 0, self.cfg.denoising_inference_steps + 1)[:-1]
        tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

        rmsds = []
        for k in range(orig_complex_graphs.num_graphs):
            data_list = [Batch.from_data_list([copy.deepcopy(orig_complex_graphs[k])])]

            randomize_position(
                data_list,
                self.cfg.diffusion.no_torsion,
                False,
                self.cfg.diffusion.tr_sigma_max,
            )

            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list is None and failed_convergence_counter <= 5:
                try:
                    predictions_list, confidences = sampling(
                        data_list=data_list,
                        model=self.net,
                        denoising_inference_steps=self.cfg.denoising_inference_steps,
                        tr_schedule=tr_schedule,
                        rot_schedule=rot_schedule,
                        tor_schedule=tor_schedule,
                        device=self.device,
                        t_to_sigma=self.net.t_to_sigma,
                        model_cfg=self.cfg,
                        batch_size=1,
                    )
                except Exception as expt:
                    if "failed to converge" in str(expt):
                        failed_convergence_counter += 1
                        self.print("SVD failed to converge - trying again with a new sample")
                    else:
                        raise expt
            if failed_convergence_counter > 5:
                self.print("SVD failed to converge 5 times - skipping the complex")
                continue

            orig_complex_graph = orig_complex_graphs[k]
            if self.cfg.diffusion.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (
                    orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                )

            filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

            if isinstance(orig_complex_graph["ligand"].orig_pos, list):
                orig_complex_graph["ligand"].orig_pos = orig_complex_graph["ligand"].orig_pos[0]

            ligand_pos = np.asarray([predictions_list[0]['ligand'].pos.cpu().numpy()[filterHs]])
            orig_ligand_pos = np.expand_dims(
                orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(),
                axis=0,
            )
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
            rmsds.append(rmsd)
        rmsds = np.array(rmsds)

        return rmsds

    def loss_function(self, tr_pred, rot_pred, tor_pred, batch, apply_mean=True):
        rot_weight = self.cfg.diffusion.rot_weight
        tor_weight = self.cfg.diffusion.tor_weight
        no_torsion = self.cfg.diffusion.no_torsion
        tr_weight = self.cfg.diffusion.tr_weight
        tr_sigma, rot_sigma, tor_sigma = self.net.t_to_sigma(
            *[
                (
                    torch.cat([d.complex_t[noise_type] for d in batch])
                    if isinstance(batch, list)
                    else batch.complex_t[noise_type]
                )
                for noise_type in ["tr", "rot", "tor"]
            ]
        )
        mean_dims = (0, 1) if apply_mean else 1
        # translation component
        tr_score = torch.cat([d.tr_score for d in batch], dim=0) if isinstance(batch, list) else batch.tr_score
        tr_sigma = tr_sigma.unsqueeze(-1)
        tr_loss = ((tr_pred - tr_score) ** 2 * tr_sigma**2).mean(dim=mean_dims)
        tr_base_loss = (tr_score**2 * tr_sigma**2).mean(dim=mean_dims).detach()
        # rotation component
        rot_score = torch.cat([d.rot_score for d in batch], dim=0) if isinstance(batch, list) else batch.rot_score
        rot_score_norm = so3_score_norm(rot_sigma).unsqueeze(-1)
        rot_loss = (((rot_pred - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims)
        rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()
        # torsion component
        if not no_torsion:
            edge_tor_sigma = torch.from_numpy(
                np.concatenate([d.tor_sigma_edge for d in batch] if isinstance(batch, list) else batch.tor_sigma_edge)
            )
            tor_score = torch.cat([d.tor_score for d in batch], dim=0) if isinstance(batch, list) else batch.tor_score
            tor_score_norm2 = torch.tensor(
                self.net.torus_score_norm(edge_tor_sigma.cpu().numpy()),
                device=self.device,
                dtype=torch.float,
            )
            tor_loss = (tor_pred - tor_score) ** 2 / tor_score_norm2
            tor_base_loss = ((tor_score**2 / tor_score_norm2)).detach()
            if apply_mean:
                tor_loss = self.empty_safe_mean(tor_loss, dim=0, keepdim=True)
                tor_base_loss = self.empty_safe_mean(tor_base_loss, dim=0, keepdim=True)
            else:
                index = (
                    torch.cat([torch.ones(d["ligand"].edge_mask.sum()) * i for i, d in enumerate(batch)]).long()
                    if isinstance(batch, list)
                    else batch["ligand"].batch[batch["ligand", "ligand"].edge_index[0][batch["ligand"].edge_mask]]
                )
                num_graphs = len(batch) if isinstance(batch, list) else batch.num_graphs
                t_l, t_b_l, c = (
                    torch.zeros(num_graphs, device=self.device),
                    torch.zeros(num_graphs, device=self.device),
                    torch.zeros(num_graphs, device=self.device),
                )
                c.index_add_(0, index, torch.ones(tor_loss.shape, device=self.device))
                c = c + 0.0001
                t_l.index_add_(0, index, tor_loss)
                t_b_l.index_add_(0, index, tor_base_loss)
                tor_loss, tor_base_loss = t_l / c, t_b_l / c
        else:
            if apply_mean:
                tor_loss, tor_base_loss = (
                    torch.zeros(1, dtype=torch.float, device=self.device),
                    torch.zeros(1, dtype=torch.float, device=self.device),
                )
            else:
                tor_loss, tor_base_loss = (
                    torch.zeros(len(rot_loss), dtype=torch.float, device=self.device),
                    torch.zeros(len(rot_loss), dtype=torch.float, device=self.device),
                )
        loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight
        return (
            loss,
            tr_loss.detach(),
            rot_loss.detach(),
            tor_loss.detach(),
            tr_base_loss,
            rot_base_loss,
            tor_base_loss,
        )

    def loss_function_confidence(self, pred, batch):
        device = pred.device
        if self.cfg.rmsd_prediction:
            labels = torch.cat([graph.rmsd for graph in batch]).to(device) if isinstance(batch, list) else batch.rmsd
            confidence_loss = torch.nn.functional.mse_loss(pred, labels)
        else:
            if isinstance(self.cfg.rmsd_classification_cutoff, ListConfig):
                labels = (
                    torch.cat([graph.y_binned for graph in batch]).to(device)
                    if isinstance(batch, list)
                    else batch.y_binned
                )
                confidence_loss = torch.nn.functional.cross_entropy(pred, labels)
            else:
                labels = torch.cat([graph.y for graph in batch]).to(device) if isinstance(batch, list) else batch.y
                confidence_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, labels)
        return confidence_loss

    @staticmethod
    def empty_safe_mean(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """If the tensor is empty, return 0 instead of nan. Otherwise apply regular mean"""
        if 0 in tensor.shape:
            new_shape = [s if s != 0 else 1 for s in tensor.shape]
            tensor = torch.zeros(
                new_shape,
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=tensor.device,
            )
        return torch.mean(tensor, **kwargs)

    @classmethod
    def list_available_models(cls) -> PretrainedModelInfo:
        return None


class DiffdockTensorProductScoreModel(TensorProductScoreModelBase):
    """
    Nemo wrapper of diffdock tensor product coarse-grained score model
    """

    def __init__(self, cfg: OmegaConf, trainer: Trainer = None, data_manager=None):
        if "model" in cfg:
            cfg_ = cfg.model
        else:
            cfg_ = cfg
        super().__init__(
            cfg=cfg_,
            trainer=trainer,
            data_manager=data_manager,
            net=TensorProductScoreModel(cfg=cfg_),
        )
        logging.info("inheriting from CG tensor product score model")


class DiffdockTensorProductScoreModelAllAtom(TensorProductScoreModelBase):
    """
    Nemo wrapper of diffdock tensor product all-atom score model
    """

    def __init__(self, cfg: OmegaConf, trainer: Trainer = None, data_manager=None):
        if "model" in cfg:
            cfg_ = cfg.model
        else:
            cfg_ = cfg
        super().__init__(
            cfg=cfg_,
            trainer=trainer,
            data_manager=data_manager,
            net=TensorProductScoreModelAllAtom(cfg=cfg_),
        )
        logging.info("inheriting from All-Atom tensor product score model")
