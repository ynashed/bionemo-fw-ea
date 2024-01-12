# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from typing import Dict, List, Optional

import torch
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch.cuda.amp import autocast

from bionemo.data.dataloader import ProteinBertCollate
from bionemo.data.dataloader.protein_collate import ESM2BertCollate
from bionemo.data.dataset_builder_utils import build_typed_dataset
from bionemo.data.mapped_dataset import NeMoUpsampling, Uniref90ClusterMappingDataset
from bionemo.data.molecule import megamolbart_build_train_valid_test_datasets
from bionemo.model.protein.esm1nv.base import ESMnvMegatronBertModel


__all__ = ["ESM1nvModel", "ESM2nvModel"]


class ESM1nvModel(ESMnvMegatronBertModel):
    """
    ESM1nv pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        """
        model_name = self._cfg.tokenizer.get('model_name')
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=model_name,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            legacy=False,
        )
        # patch tokenizer for use with HF esm tokenizer
        if self._cfg.tokenizer.library == 'huggingface' and str(model_name).startswith('facebook/esm2'):
            self.tokenizer.tokenizer.vocab = self.tokenizer.tokenizer.get_vocab()

    def build_pretraining_data_loader(self, dataset, consumed_samples, num_workers=None):
        """Buld dataloader given an input dataset."""

        assert self._cfg.data.dataloader_type == 'single', AssertionError(
            f'Only the Megatron sequential ("single") sampler is currently supported. {self._cfg.data.dataloader_type} was chosen.'
        )

        from megatron.core import parallel_state
        from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
            MegatronPretrainingSampler,
        )

        # NOTE (SKH) this was taken directly from megatron, this is the 'single' dataloader type.
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.cfg.micro_batch_size,
            global_batch_size=self.cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=self.cfg.get('drop_last', True),
        )

        if num_workers is None:
            num_workers = self.cfg.data.num_workers
        # Torch dataloader.
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,  # Needs to be set to zero.
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        dataloader.pin_memory = False  # must be False with CSV dataset TODO check with binary
        pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False
        if self.cfg.pipeline_model_parallel_size > 1 and self.cfg.data.dynamic_padding:
            raise ValueError("Pipeline model parallelism does not support dynamic_padding.")
        dataloader.collate_fn = ProteinBertCollate(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            modify_percent=self._cfg.data.modify_percent,
            perturb_percent=self._cfg.data.perturb_percent,
            dynamic_padding=self.cfg.data.dynamic_padding,
        ).collate_fn

        return dataloader

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            if self._validation_ds is not None:
                consumed_samples = 0
                self._validation_dl = self.build_pretraining_data_loader(
                    self._validation_ds, consumed_samples, num_workers=0
                )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            if self._test_ds is not None:
                consumed_samples = 0
                self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples, num_workers=0)

    @staticmethod
    def _build_train_valid_test_datasets(trainer, model_cfg):
        logging.info('Building Bert datasets.')
        global_batch_size = trainer.world_size * model_cfg.micro_batch_size / model_cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = trainer.max_steps * trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // trainer.val_check_interval + 1) * trainer.limit_val_batches
        test_iters = trainer.limit_test_batches

        train_valid_test_num_samples = {
            'train': int(max_train_steps * global_batch_size),
            'val': int(eval_iters * global_batch_size),
            'test': int(test_iters * global_batch_size),
        }

        _train_ds, _validation_ds, _test_ds = megamolbart_build_train_valid_test_datasets(
            cfg=model_cfg.data, train_valid_test_num_samples=train_valid_test_num_samples
        )

        logging.info(f'Length of train dataset: {len(_train_ds)}')
        logging.info(f'Length of val dataset: {len(_validation_ds)}')
        logging.info(f'Length of test dataset: {len(_test_ds) if _test_ds is not None else None}')
        logging.info('Finished building Bert datasets.')
        return _train_ds, _validation_ds, _test_ds

    def build_train_valid_test_datasets(self):
        train, val, test = self._build_train_valid_test_datasets(self.trainer, self._cfg)
        self._train_ds = train
        self._validation_ds = val
        self._test_ds = test
        return self._train_ds, self._validation_ds, self._test_ds

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        """The function computes and logs three scores:
            - The average cross entropy loss over the validation data
            - The exponential of the averaged loss
            - The average perplexity score over the validation data.

        The estimate of the perplexity is defined as the exponential of the average of the masked CE losses: `exp(-loss_mean)`

        Args:
            outputs: A list of dictionaries, where each dictionary represents the output of a validation step.
            The computed values are logged using the Lightning logger.
        """
        if not outputs:
            return
        averaged_loss = torch.stack(outputs).mean()
        average_perplexity = averaged_loss.exp()
        self.log('val_loss', averaged_loss, prog_bar=True)
        self.log('val_perplexity', average_perplexity)
        self.log('val_loss_ECE', pow(2, averaged_loss))  # calculate exponential cross entropy loss for logs
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step - self.init_global_step))

    def encode(self, tokens_enc, enc_mask):
        # FIXME this autocast shouldn't be needed
        with autocast(enabled=self.enable_autocast):
            hidden_states = self(tokens_enc, enc_mask, None)
            if self.model.post_process:
                hidden_states = hidden_states[0]
        return hidden_states

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        """
        The function computes and logs three scores:
            - the average cross-entropy loss over the test data,
            - the exponential of the averaged loss, and
            - the average perplexity score over the test data.

        The estimate of the perplexity is defined as the exponential of the average of the masked CE losses: `exp(-loss_mean)`
        This function is called at the end of the testing step `model.test()`.

        Args:
            outputs: A list of dictionaries, where each dictionary represents the output of a validation step.
            The computed values are logged using the NeMo logger.
        """
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')
        logging.info(f'test_loss_ECE: {pow(2, averaged_loss[0])}')
        logging.info(f'test_perplexity: {averaged_loss[0].exp()}')

    @property
    def input_names(self):
        return [
            'input_ids',
            'attention_mask',
        ]

    @property
    def output_names(self):
        return ['output']

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {'input_ids': {0: 'batch', 1: 'time'}, 'attention_mask': {0: 'batch', 1: 'time'}}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {'output': {0: 'batch', 1: 'time', 2: 'size'}}


class ESM2nvModel(ESM1nvModel):
    """
    Extends the ESM1nv model by customizing dataset construction to mimic ESM2's implementation.

    This model introduces a dataset structure that not only retains elements from ESM1nv but also
    incorporates new fields specifically tailored for ESM2nv. One significant change involves using
    NeMoUpsampling to upsample UniRef50 cluster IDs and pre-computing samples from UniRef90 via a mapped dataset.

    The configuration files are designed to incorporate specific argument structures. A sample configuration
    demonstrating the dataset configuration can be viewed in `examples/protein/esm2nv/conf/base_config.yaml`

    Args:
        cfg (DictConfig): The configuration object, typically constructed from a YAML file.
        trainer (Trainer): The training instance associated with this model.

    Attributes:
        dataset_path (str): Parent directory containing train, test, and validation data.
        dataset (dict): Specifies the data file range for training, testing, and validation.
        data_impl (str): Implementation choice, like "csv_mmap".
        data_impl_kwargs (dict): Arguments specific to the data implementation choice.
        uf50_datapath (str): Path to the raw UniRef50 fasta file.
        uf90_datapath (str): Path to the raw UniRef90 fasta file.
        cluster_mapping_tsv (str): TSV file mapping UniRef50 cluster IDs to UniRef90 entries.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

    @staticmethod
    def _build_train_valid_test_datasets(trainer, model_cfg, keep_uf50=False):
        '''
        keep_uf50 - use this flag for testing, manually assigns the uniref50_dataset object to the resulting Uniref90ClusterMappingDataset object.
        '''
        _train_ds, _validation_ds, _test_ds = ESM1nvModel._build_train_valid_test_datasets(trainer, model_cfg)

        dataset_path = model_cfg.data.uf90.uniref90_path
        split = 'uf90_csvs'
        ds_name = model_cfg.data.uf90.dataset.get(split, None)
        filepath: str = os.path.join(dataset_path, split, ds_name)

        data_impl = model_cfg.data.uf90.get('data_impl', None)
        assert data_impl is not None, 'Config "cfg" should contain field "cfg.data_impl"'
        # NOTE train/val/test will all each into uniref90, since we are split on clusters, we now they are independent.
        #   hence, we only need one dataset object for uf90
        uniref90_dataset = build_typed_dataset(
            dataset_paths=filepath,
            data_impl=data_impl,
            cfg=model_cfg.data.uf90,
            use_upsampling=False,
            num_samples=None,
        )

        results = []
        for ds, split in zip([_train_ds, _validation_ds, _test_ds], ['train', 'val', 'test']):
            # TypeHint for intellisense
            _ds: NeMoUpsampling = ds

            index_mapping_dir = ds.index_mapping_dir

            path_root: str = os.path.join(model_cfg.data.dataset_path, split)
            # Setup the resampling
            ds = Uniref90ClusterMappingDataset(
                uniref50_dataset=_ds,
                uniref90_dataset=uniref90_dataset,
                data_prefix=split,  # used for index creation
                seed=model_cfg.seed,  # used for rng, although awkward because global statehood
                index_mapping_dir=index_mapping_dir,  # stores index
                cluster_map_starts_fn=f'{path_root}/starts.mmap',
                cluster_map_counts_fn=f'{path_root}/counts.mmap',
                name=ds.name,
            )

            if keep_uf50:
                ds.uniref50_dataset = _ds

            results.append(ds)

        [_train_ds, _validation_ds, _test_ds] = results
        logging.info(f'Length of train dataset: {len(_train_ds)}')
        logging.info(f'Length of val dataset: {len(_validation_ds)}')
        logging.info(f'Length of test dataset: {len(_test_ds)}')
        logging.info('Finished building Bert datasets.')
        return _train_ds, _validation_ds, _test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples, num_workers=None):
        """Buld dataloader given an input dataset."""

        assert self._cfg.data.dataloader_type == 'single', AssertionError(
            f'Only the Megatron sequential ("single") sampler is currently supported. {self._cfg.data.dataloader_type} was chosen.'
        )
        dataloader = super().build_pretraining_data_loader(
            dataset=dataset, consumed_samples=consumed_samples, num_workers=num_workers
        )

        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        dataloader.pin_memory = False

        pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False
        if self.cfg.pipeline_model_parallel_size > 1 and self.cfg.data.dynamic_padding:
            raise ValueError("Pipeline model parallelism does not support dynamic_padding.")
        dataloader.collate_fn = ESM2BertCollate(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            modify_percent=self._cfg.data.modify_percent,
            perturb_percent=self._cfg.data.perturb_percent,
            dynamic_padding=self.cfg.data.dynamic_padding,
        ).collate_fn
        return dataloader
