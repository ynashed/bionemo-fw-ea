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

from nemo.collections.nlp.models.language_modeling.megatron_bert_model import (
    MegatronBertModel
)
from nemo.collections.nlp.modules.common.tokenizer_utils import (
    get_nmt_tokenizer
)
from bionemo.data.dataloader.kmer_collate import DeterministicLengthTruncator
from bionemo.tokenizer import KmerTokenizer
from bionemo.data.dataloader import (
    KmerBertCollate,
    SpanMasking,
    LengthTruncator,
    KmerTokenizerAdapter,
)
import torch

from nemo.utils import logging
import os

from bionemo.data.utils import (
    build_train_valid_test_datasets,
    FormattedDatasetFactory,
    DatasetBuilderSpec,
    expand_dataset_paths,
)
from bionemo.data.fasta_dataset import ConcatFastaDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

# TODO: is HAVE_APEX needed?
try:
    from apex.transformer import tensor_parallel

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = [
    "tokenizers",
    "adapters",
    "DNABERTDatasetFactory",
    "DNABERTModel",
    "FastaDatasetBuilder",
]


tokenizers = {
    'kmer': KmerTokenizer,
}

adapters = {
    'kmer': KmerTokenizerAdapter,
}


class FastaDatasetBuilder(DatasetBuilderSpec):

    def format_dataset_paths(self):
        """
        Parses FASTA paths.

        """
        self.dataset_paths = expand_dataset_paths(
            self.options['filepath'], None)

    def check_path(self, filepath):
        """
        Checks whether a FASTA exists.

        Arguments:
            filepath (str): a string that can be used to identify the filepath

        Returns:
            Optional[str]: If the file exists, this returns None, otherwise
                it returns the on the filepath.

        """
        if not os.path.exists(filepath):
            return filepath

    def create_dataset(self):
        """
        Instantiates a FastaDataset.

        Returns:
            Dataset: Dataset instantiated from paths.
        """
        cfg = self.options['cfg']
        max_length = cfg.seq_length - 1 + cfg.k
        self.dataset = ConcatFastaDataset(
            self.dataset_paths, max_length, backend='memory',
            )
        return self.dataset


class DNABERTDatasetFactory(FormattedDatasetFactory):

    def __init__(self):
        """
        Initializes a dataset factory for handling fasta formats.
        """
        self.formats = {
            'fasta': FastaDatasetBuilder,
        }


class DNABERTModel(MegatronBertModel):
    """
    WIP class for pre-training DNABERT models

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO, should we do something like extract a superlcass and move
        # up the pad_size attribute?
        self.pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        """
        tokenizer_type = self._cfg.tokenizer.type
        if tokenizer_type in tokenizers:
            self.tokenizer = KmerTokenizerAdapter(
                tokenizers[tokenizer_type].from_vocab_file(
                    self._cfg.tokenizer.model,
                    self._cfg.tokenizer.vocab_file,
                ))
        else:
            self.tokenizer = get_nmt_tokenizer(
                library=self._cfg.tokenizer.library,
                model_name=tokenizer_type,
                tokenizer_model=self.register_artifact(
                    "tokenizer.model",
                    self._cfg.tokenizer.model
                    ),
                vocab_file=self.register_artifact(
                    "tokenizer.vocab_file",
                    self._cfg.tokenizer.vocab_file
                    ),
                legacy=False,
            )

    @staticmethod
    def _get_random_length_truncator():
        sentence_transform = LengthTruncator()
        sentence_transform.get_sentence = lambda x: x['seq']
        return sentence_transform

    @staticmethod
    def _get_deterministic_length_truncator():
        sentence_transform = DeterministicLengthTruncator()
        sentence_transform.get_sentence = lambda x: x['seq']
        return sentence_transform

    def _setup_collate(self, dataloader, sentence_transform):
        dataloader.collate_fn = KmerBertCollate(
            self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=self.pad_size_divisible_by_8,
            masking_strategy=SpanMasking(
                tokenizer=self.tokenizer,
                seed_probability=0.15,
                span_length=self._cfg.tokenizer.k,
            ),
            transform_sentences=sentence_transform,
        ).collate_fn

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)
        dataloader = self._train_dl
        sentence_transform = self._get_random_length_truncator()
        self._setup_collate(dataloader, sentence_transform)

    def setup_validation_data(self, cfg):
        super().setup_validation_data(cfg)
        dataloader = self._validation_dl
        sentence_transform = self._get_deterministic_length_truncator()
        self._setup_collate(dataloader, sentence_transform)

    def setup_test_data(self, cfg):
        super().setup_test_data(cfg)
        dataloader = self._test_dl
        sentence_transform = self._get_deterministic_length_truncator()
        self._setup_collate(dataloader, sentence_transform)

    @classmethod
    def list_available_models(cls):
        """
        TODO add documentation

        This overrides a functionality from NeMo that lists pre-trained models.
        We don't have any yet.
        """
        return []

    # TODO the functions after this are _highly_ similar to the code
    # in ESM model, we should regroup it:
    def _build_train_valid_test_datasets(self):
        logging.info('Building Bert datasets.')
        global_batch_size = self.trainer.world_size * self._cfg.micro_batch_size / self._cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            int (max_train_steps * global_batch_size),
            int (eval_iters * global_batch_size),
            int (test_iters * global_batch_size),
        ]

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self._cfg.data,
            trainer=self.trainer,
            train_valid_test_num_samples=train_valid_test_num_samples,
            dataset_factory=DNABERTDatasetFactory(),
        )

        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building Bert datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def process_batch(self, batch):
        """Build the batch."""
        # Items and their type.
        keys = ['tokens', 'labels', 'loss_mask', 'padding_mask']

        datatype = torch.int64
        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens = data_b['tokens'].long()
        loss_mask = data_b['loss_mask'].float()
        lm_labels = data_b['labels'].long()
        padding_mask = data_b['padding_mask'].long()
        types = None #expected by training & validation methods
        sentence_order = None #expected by training & validation methods
        return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
