# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import math
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.utils import logging

from bionemo.data.prott5_utils import prott5_build_dataset

from nemo.collections.nlp.parts.nlp_overrides import (
    NLPSaveRestoreConnector,
)

class T5SaveRestoreConnector(NLPSaveRestoreConnector):
    # 128 -- is the number of padded vocabulary in MegatronT5Model
    def __init__(self, vocab_size=128) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def modify_state_dict(self, conf, state_dict):
        new_state_dict = {}
        # trunace the word_embeddings and tokens_head
        for key in state_dict.keys():
            if ("word_embeddings" in key) or ("tokens_head" in key):
                # initialize with pretrained word embeddings 
                token_embeddings = state_dict[key]
                logging.info(f"Updating key={key}, token_embeddings.shape={token_embeddings.shape}, vocab_size={self.vocab_size}")
                # tile token_embeddings to be at least self.vocab_size
                dims = (math.ceil(self.vocab_size / token_embeddings.shape[0]),)
                # we want to tile only first dimension
                if len(token_embeddings.shape) == 2:
                    dims += (1,)
                token_embeddings = token_embeddings.tile(dims=dims)
                new_state_dict[key] = token_embeddings[:self.vocab_size]
            elif key.endswith("encoder_embedding.position_embeddings.weight") or key.endswith("decoder_embedding.position_embeddings.weight"):
                position_embeddings = state_dict[key]
                # allow changing the position embeddings for learned_abs
                if "encoder_embedding" in key:
                    if "encoder" in conf:
                        max_position_embeddings = conf.encoder.max_position_embeddings
                    else:
                        max_position_embeddings = conf.max_position_embeddings
                else:
                    if "decoder" in conf:
                        max_position_embeddings = conf.decoder.max_position_embeddings
                    else:
                        max_position_embeddings = conf.max_position_embeddings
                logging.info(f"Updating key={key}, position_embeddings.shape={position_embeddings.shape}, max_position_embeddings={max_position_embeddings}")
                # tile position_embeddings to be at least max_position_embeddings
                position_embeddings = position_embeddings.tile((math.ceil(max_position_embeddings / position_embeddings.shape[0]), 1))
                new_state_dict[key] = position_embeddings[:max_position_embeddings]
            else:
                new_state_dict[key] = state_dict[key]

        state_dict = new_state_dict
        return state_dict


class ProtT5nvModel(MegatronT5Model):
    """
    Prot T5 training
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        # validate cfg
        self._validate_cfg()

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)

    def setup_validation_data(self, cfg):
        super().setup_validation_data(cfg)
        if hasattr(self, '_validation_dl') and self._validation_dl is not None:
            self._validation_dl.num_workers = 0

    def setup_test_data(self, cfg):
        super().setup_test_data(cfg)
        if hasattr(self, '_test_dl') and self._test_dl is not None:
            self._test_dl.num_workers=0

    def build_train_valid_test_datasets(self):
        logging.info(f'Building {self.model_name} datasets.')
        global_batch_size = self._cfg.global_batch_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        kwargs = {
            "cfg": self._cfg,
            "trainer": self.trainer,
            "tokenizer": self.tokenizer,
            "data_impl": self._cfg.data.data_impl,
            "max_seq_length": self._cfg.data.seq_length,
            "max_seq_length_dec": self._cfg.data.seq_length_dec,
            "masked_lm_prob": self._cfg.data.masked_lm_prob,
            "short_seq_prob": self._cfg.data.short_seq_prob,
            "seed": self._cfg.seed,
            "skip_warmup": self._cfg.data.skip_warmup,
            "dataset_type": self._cfg.data.get('dataset_type', self.model_name.lower()),
            "max_ngram_size": self._cfg.data.get('max_ngram_size', 1),
            "mean_ngram_size": self._cfg.data.get('mean_ngram_size', None),
            "geometric_dist": self._cfg.data.get('geometric_dist', True),
            "permutation": self._cfg.data.get('permutation', False),
            "whole_word_masking": self._cfg.data.get('whole_word_masking', False),
            "favor_long_ngrams": self._cfg.data.get('favor_long_ngrams', False),
            "data_impl_kwargs": self._cfg.data.get('data_impl_kwargs', {})
        }

        dataset_path = self._cfg.data.dataset_path
        ds_train = self._cfg.data.dataset.train
        ds_val = self._cfg.data.dataset.val
        ds_test = self._cfg.data.dataset.test
        self._train_ds = prott5_build_dataset(
            data_prefix=os.path.join(dataset_path, 'train', ds_train),
            num_samples=train_valid_test_num_samples[0],
            name="train",
            **kwargs
            )
        self._validation_ds = prott5_build_dataset(
            data_prefix=os.path.join(dataset_path, 'val', ds_val),
            num_samples=train_valid_test_num_samples[1],
            name="valid",
            **kwargs
            )

        self._test_ds = prott5_build_dataset(
            data_prefix=os.path.join(dataset_path, 'test', ds_test),
            num_samples=train_valid_test_num_samples[2],
            name="test",
            **kwargs
            )

        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building {self.model_name} datasets.')
        return self._train_ds, self._validation_ds, self._test_ds
