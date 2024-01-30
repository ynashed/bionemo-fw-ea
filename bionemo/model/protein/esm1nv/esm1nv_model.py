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
import os
from typing import Dict, List, Optional, Union

import torch
from megatron.core import parallel_state
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core import Dataset
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch.cuda.amp import autocast

from bionemo.data.dataloader import ProteinBertCollate
from bionemo.data.dataloader.protein_collate import ESM2BertCollate
from bionemo.data.dataset_builder_utils import build_typed_dataset
from bionemo.data.mapped_dataset import Uniref90ClusterMappingDataset
from bionemo.model.protein.esm1nv.base import ESMnvMegatronBertModel


__all__ = ["ESM1nvModel", "ESM2nvModel"]


def esm1nv_build_train_valid_test_datasets(
    cfg: DictConfig, train_valid_test_num_samples: Dict[str, Optional[int]]
) -> List[Dataset]:
    """
    Build train, validation and test for pretraining of MegaMolBartModel.
    Args:
        cfg: config of data components
        train_valid_test_num_samples: dict that specifies split-specific size of loaded dataset
    Returns:
        list of dataset for splits
    """
    cfg = copy.deepcopy(cfg)

    # setting
    use_upsampling: bool = cfg.get('use_upsampling', True)
    data_impl: str = cfg.get('data_impl', None)
    # assert data_impl is not None, 'Config "cfg" should contain field "cfg.data_impl"'
    dataset_path: str = cfg.get('dataset_path', None)
    assert dataset_path is not None, 'Config "cfg" should contain field "cfg.dataset_path"'

    assert all(
        split in ['train', 'val', 'test'] for split in train_valid_test_num_samples.keys()
    ), 'Incorrect key in train_valid_test_num_samples!'

    datasets = []
    # Build individual datasets.
    for split in train_valid_test_num_samples.keys():
        num_samples = train_valid_test_num_samples[split]
        print(f'{split}:{num_samples}')
        if num_samples is None or num_samples > 0:
            ds_name: Optional[Union[str, List[Union[int, str]]]] = cfg.dataset.get(split, None)
            assert ds_name is not None, (
                f'Config "cfg" should contain field "cfg.dataset.{split}" with name or list of '
                f'names corresponding to the data files used to construct the dataset'
            )
            filepath: str = os.path.join(dataset_path, split, ds_name)
            dataset = build_typed_dataset(
                dataset_paths=filepath,
                data_impl=data_impl,
                use_upsampling=use_upsampling if num_samples is not None else False,
                cfg=cfg,
                num_samples=num_samples,
            )
        else:
            dataset = None
        datasets.append(dataset)  # These can be my train/ val/ test datasets.

    return datasets


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
        # Compute training micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        # limit_val_batches specifies the number of batches.
        max_train_steps = trainer.max_steps * trainer.accumulate_grad_batches

        eval_iters = (max_train_steps // trainer.val_check_interval + 1) * trainer.limit_val_batches
        test_iters = trainer.limit_test_batches

        train_valid_test_num_samples = {
            'train': int(max_train_steps * global_batch_size),
            'val': int(eval_iters * global_batch_size),
            'test': int(test_iters * global_batch_size),
        }

        # Note(@jomitchell) ESM should not be calling megamolbart's dataset builder.
        _train_ds, _validation_ds, _test_ds = esm1nv_build_train_valid_test_datasets(
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

    def on_validation_epoch_end(self) -> None:
        """The function computes and logs three scores:
            - The average cross entropy loss over the validation data
            - The exponential of the averaged loss
            - The average perplexity score over the validation data.

        The estimate of the perplexity is defined as the exponential of the average of the masked CE losses: `exp(-loss_mean)`

        Args:
            outputs: A list of dictionaries, where each dictionary represents the output of a validation step.
            The computed values are logged using the Lightning logger.
        """
        outputs = self.validation_step_outputs
        if not outputs:
            return
        averaged_loss = torch.stack(outputs).mean()
        average_perplexity = averaged_loss.exp()
        self.log('val_loss', averaged_loss, prog_bar=True)
        self.log('val_perplexity', average_perplexity)
        self.log('val_loss_ECE', pow(2, averaged_loss))  # calculate exponential cross entropy loss for logs
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step - self.init_global_step))
        self.validation_step_outputs.clear()

    def encode(self, tokens_enc, enc_mask):
        # FIXME this autocast shouldn't be needed
        with autocast(enabled=self.enable_autocast):
            hidden_states = self(tokens_enc, enc_mask, None)
            if self.model.post_process:
                hidden_states = hidden_states[0]
        return hidden_states

    def on_test_epoch_end(self) -> None:
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
        averaged_loss = average_losses_across_data_parallel_group(self.test_step_outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')
        logging.info(f'test_loss_ECE: {pow(2, averaged_loss[0])}')
        logging.info(f'test_perplexity: {averaged_loss[0].exp()}')
        self.test_step_outputs.clear()

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

    def build_train_dataset(
        self,
        model_cfg: DictConfig,
        num_samples: Optional[int] = None,
    ) -> Uniref90ClusterMappingDataset:
        """Constructs a train dataset.

        Args:
            num_samples: The number of samples in the dataset
            model_cfg: A config file that contains certain keys that tell us:
                1. Where the UF50 dataset lies
                2. Where the UF90 dataset lies
                3. How we want to map between the two etc.
        Returns:
            train_dataset: A UF90 cluster mapping train dataset.

        # TODO(@jomitchell) Enable creation of train dataset from single .fasta file.
        """
        # Create the training dataset
        train_ds = build_typed_dataset(
            dataset_paths=os.path.join(model_cfg.data.train.dataset_path, 'train', model_cfg.data.train.range),
            data_impl=model_cfg.data.train.data_impl,
            cfg=model_cfg.data.train,
            use_upsampling=model_cfg.data.train.use_upsampling,
            num_samples=num_samples,
        )

        uniref90_dataset = build_typed_dataset(
            dataset_paths=os.path.join(
                model_cfg.data.train.uf90.uniref90_path, 'uf90_csvs', model_cfg.data.train.range
            ),
            data_impl=model_cfg.data.train.uf90.data_impl,
            cfg=model_cfg.data.train.uf90,
            use_upsampling=False,  # TODO(@jomitchell): Jasleen can these be upsampled?
            num_samples=None,
        )

        index_mapping_dir = train_ds.index_mapping_dir
        # TODO(@jomitchell): Dataset refactor: Path to mmap_dataset should be full path to `train`
        mmap_dataset_path: str = os.path.join(model_cfg.data.train.dataset_path, "train")
        # Setup the resampling
        train_dataset = Uniref90ClusterMappingDataset(
            uniref50_dataset=train_ds,
            uniref90_dataset=uniref90_dataset,
            data_prefix="train",  # used for index creation
            seed=model_cfg.seed,  # used for rng, although awkward because global statehood
            index_mapping_dir=index_mapping_dir,  # stores index
            cluster_map_starts_fn=f'{mmap_dataset_path}/starts.mmap',
            cluster_map_counts_fn=f'{mmap_dataset_path}/counts.mmap',
            name=train_ds.name,
        )
        return train_dataset

    def build_val_dataset(
        self,
        model_cfg: DictConfig,
        num_samples: Optional[int] = None,
        limit_batches_scale_factor: Optional[float] = None,
    ):
        # TODO: If `num_samples is None` do we load the full dataset?
        # IF not default none -> raise warning.
        val_ds = build_typed_dataset(
            dataset_paths=os.path.join(model_cfg.data.val.dataset_path, 'val', model_cfg.data.val.range),
            data_impl=model_cfg.data.val.data_impl,
            cfg=model_cfg.data.val,
            use_upsampling=model_cfg.data.val.use_upsampling,
            num_samples=num_samples,
            limit_batches_scale_factor=limit_batches_scale_factor,
        )
        return val_ds

    def build_test_dataset(
        self,
        model_cfg: DictConfig,
        num_samples: Optional[int] = None,
        limit_batches_scale_factor: Optional[float] = None,
    ):
        """Constructs the test dataset."""
        test_ds = build_typed_dataset(
            dataset_paths=os.path.join(model_cfg.data.test.dataset_path, 'test', model_cfg.data.test.range),
            data_impl=model_cfg.data.test.data_impl,
            cfg=model_cfg.data.test,
            use_upsampling=model_cfg.data.test.use_upsampling,
            num_samples=num_samples,
            limit_batches_scale_factor=limit_batches_scale_factor,
        )
        return test_ds

    def build_train_valid_test_datasets(self):
        if self.trainer.limit_test_batches == 0:
            raise ValueError("trainer.limit_test_batches is set to 0 which means you will have no test data.")

        if self.trainer.limit_val_batches == 0:
            raise ValueError(
                "trainer.limit_val_batches is set to 0 which means you will have no val data."
                "Please use fractional values 0<x<1 to get fractional data, or 1 for full data."
                "Or an int > 1 to specify the exact number of batches that you want."
            )

        # If val upsampling is false, and we expect to be up/ downsampling val.
        if not self._cfg.data.val.use_upsampling and self.trainer.limit_val_batches != 1.0:
            raise ValueError(
                f"config.model.data.val.use_upsampling is {self._cfg.data.val.use_upsampling} but self.trainer.limit_val_batches is: "
                f"{self.trainer.limit_val_batches} which is not 1.0"
            )

        # If test upsampling is false, and we expect to be downsampling test.
        if not self._cfg.data.test.use_upsampling and self.trainer.limit_test_batches != 1.0:
            raise ValueError(
                f"config.model.data.test.use_upsampling is {self._cfg.data.test.use_upsampling} but self.trainer.limit_test_batches is: {self.trainer.limit_test_batches}"
                "which is not equal to 1.0"
            )

        global_batch_size = self.trainer.world_size * self._cfg.micro_batch_size / self._cfg.tensor_model_parallel_size
        # Compute training micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        num_train_samples = int(max_train_steps * global_batch_size)

        # Num_val_runs is the number of times we will run validation.
        num_val_runs = max_train_steps // self.trainer.val_check_interval + 1

        limit_val_batches_scale_factor = None

        # If we want to use a set number of batches during validation, and that number is > 1.
        if isinstance(self.trainer.limit_val_batches, int):
            eval_iters = num_val_runs * self.trainer.limit_val_batches
            num_val_samples = int(eval_iters * global_batch_size)
        # If we use a float for a fractional piece of the dataset. Then we need to scale the number of iterations accordingly
        elif isinstance(self.trainer.limit_val_batches, float):
            # Note: we need to do the equation:
            # total_val_iterations = (len(dataset) * limit_val_batches) / global_batch_size but we don't have len(dataset) until further downstream.
            # So we are going to save the scale factor here and multiply by len(dataset) * limit_val_batches_scale_factor later on.
            total_val_iterations = 1.0 / global_batch_size
            eval_iters = num_val_runs * total_val_iterations
            limit_val_batches_scale_factor = self.trainer.limit_val_batches
            num_val_samples = eval_iters * global_batch_size
        else:
            raise ValueError(
                f"self.trainer.limit_val_batches is of type {type(self.trainer.limit_val_batches)}"
                "which is not supported."
            )

        limit_test_batches_scale_factor = None
        if isinstance(self.trainer.limit_test_batches, int):
            test_iters = self.trainer.limit_test_batches
            num_test_samples = int(test_iters * global_batch_size)
        elif isinstance(self.trainer.limit_test_batches, float):
            limit_test_batches_scale_factor = self.trainer.limit_test_batches
            num_test_samples = 1
        else:
            raise ValueError(
                f"self.trainer.limit_test_batches is of type {type(self.trainer.limit_test_batches)}"
                "which is not supported."
            )

        self._train_ds = self.build_train_dataset(model_cfg=self._cfg, num_samples=num_train_samples)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')

        self._validation_ds = self.build_val_dataset(
            model_cfg=self._cfg,
            num_samples=num_val_samples,
            limit_batches_scale_factor=limit_val_batches_scale_factor,
        )
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')

        self._test_ds = self.build_test_dataset(
            model_cfg=self._cfg,
            num_samples=num_test_samples,
            limit_batches_scale_factor=limit_test_batches_scale_factor,
        )
        logging.info(f'Length of test dataset: {len(self._test_ds)}')

        return self._train_ds, self._validation_ds, self._test_ds

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

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = self.build_pretraining_data_loader(
                self._validation_ds, consumed_samples, num_workers=0
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples, num_workers=0)
