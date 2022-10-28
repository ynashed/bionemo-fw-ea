from curses.ascii import SP
from bionemo.data.validation_dataset import DataFrameTransformDataset
from bionemo.data.fasta_dataset import ConcatFastaDataset
from bionemo.data import KmerBertCollate
from bionemo.tokenizer import KmerTokenizer
from functools import partial
import pandas as pd
import math

from pathlib import Path
import os

dir_ = Path('/workspace/bionemo/examples/dna/')
sampled_data_file = str(dir_ / 'data/splice-site-prediction/sampled-data-10k/sampled-data.csv')
df = pd.read_csv(sampled_data_file)
fa_template = os.path.join(str(dir_),'data/splice-site-prediction/GRCh38.ensembl.99/Homo_sapiens.GRCh38.dna.chromosome.{}.fa.gz')
chrs = list(str(i) for i in range(1, 23))

fasta_dataset = ConcatFastaDataset([fa_template.format(chr) for chr in chrs], 400, backend='file')

def get_start_end(coord, length):
    start = int(coord - math.ceil(length / 2))
    end = int(coord + math.floor(length / 2))
    return start, end


fa_tokenizer = fa_template.format(22)
tokenizer_path = '/workspace/bionemo/models/dna/dnabert/vocab/dnabert3-chr22.model'

tokenizer = KmerTokenizer.from_vocab_file(tokenizer_path)

bert_prep = KmerBertCollate(
    tokenizer,
    modify_percent=0,
    seq_length=512,
    pad_size_divisible_by_8=True,
).collate_fn


def fetch_bert_dna(row: pd.Series, length):
    mid = row.coord
    start, end = get_start_end(mid, length)
    text = fasta_dataset.fetch(row.id, start, end)
    return {key: value[0] for key, value in bert_prep([text]).items()}

def get_target(row: pd.Series):
    return {'target': row.kind}

gff_dataset = DataFrameTransformDataset(
    sampled_data_file,
    functions = [
        partial(fetch_bert_dna, length=400),
        get_target,
    ],
    read_csv_args={'dtype': {'id': str}}
    )


from bionemo.model.utils import setup_trainer
from bionemo.model.dnabert import DNABERTModel
from omegaconf import OmegaConf
from nemo.utils.app_state import AppState

import torch
import os
from nemo.utils import logging

base_cfg_file = '/workspace/bionemo/examples/dna/conf/dnabert_base_config.yaml'
cfg_file = '/workspace/bionemo/examples/dna/conf/dnabert_config_splice_site.yaml'

base_cfg = OmegaConf.load(base_cfg_file)
cfg = OmegaConf.load(cfg_file)
cfg = OmegaConf.merge(base_cfg, cfg)

cfg.trainer['strategy'] = None
cfg.exp_manager.wandb_logger_kwargs.notes = ''
cfg.exp_manager.wandb_logger_kwargs.offline = True
cfg.model.tokenizer.k = 3
ckpt = '/workspace/bionemo/examples/dna/nemo_experiments/dnabert/2022-10-25_00-04-34/checkpoints/dnabert--val_loss=8.57-step=760400-consumed_samples=3041600.0-last.ckpt'
cfg.trainer.max_steps = 5000 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches

gff_csv = '/workspace/bionemo/examples/dna/data/splice-site-prediction/sampled-data-10k/sampled-data.csv'
cfg.model.data['train_file'] = gff_csv
cfg.model.data['val_file'] = gff_csv
cfg.model.data['fasta_directory'] = '/workspace/bionemo/examples/dna/data/splice-site-prediction/GRCh38.ensembl.99'
cfg.model.data['fasta_pattern'] = 'Homo_sapiens.GRCh38.dna.chromosome.{}.fa.gz'
cfg.model.num_workers = 1

trainer = setup_trainer(cfg)

from splice_site_data_module import SpliceSiteDataModule

from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable

from bionemo.model.core import MLPModel
from torch import nn
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from apex.transformer import parallel_state

class EncoderFineTuning(ModelPT, Exportable):
    pass

class SpliceSiteBERTPredictionModel(ModelPT, Exportable):
    def compute_consumed_samples(self, steps_since_resume=0):
        app_state = AppState()
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume
            * app_state.data_parallel_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def list_available_models(self):
        return []

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = get_params_for_weight_decay_optimization(
            [self.encoder_model, self.task_head])

    def on_train_start(self) -> None:
        super().on_train_start()
        self.init_global_step = self.trainer.global_step

    def modify_encoder_model(self, encoder_model):
        # TODO maybe warn that we are turning of the post_processing
        # Also...this could be a little hacky
        encoder_model.model.post_process = False
        # TODO: make freeze encoder configurable?
        # encoder_model.freeze()
        # TODO, do we want to have ctx manager or something so we can reverse
        # the changes we make here?

    def setup_encoder_model(self, cfg, trainer):
        # TODO this could be refactored to instantiate a new model if no
        # checkpoint is specified

        model = DNABERTModel.load_from_checkpoint(
            # TODO grab ckpt from config
            ckpt,
            cfg=cfg,
            trainer=trainer,
        )
        self.modify_encoder_model(model)

        return model

    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer=trainer)
        if trainer.accumulate_grad_batches != 1:
            raise ValueError(
                "Trainer.accumulate_grad_batches currently only supported"
                " for Trainer.accumulate_grad_batches = 1")
        self.cfg = cfg

        self.encoder_model = self.setup_encoder_model(cfg, trainer)

        # TODO make number_of_classes (3) configurable for classificaiton
        # TODO make other NN MLP parameters configurable
        # TODO make MLPModel configurable
        self.task_head = MLPModel(layer_sizes=[cfg.hidden_size, 3], dropout=0.1)

        # TODO double check that this index is the correct one (according to get mid point function)
        # TODO and make it based off of the sequence length
        self.extract_for_task_head = partial(self.get_hiddens_for_idx, idx=200)
        # TODO make the loss configurable
        self.loss_fn = nn.CrossEntropyLoss() #define a loss function
        # TODO make target name configurable from cfg?
        self.batch_target_name = 'target'
        # TODO this must change to resume training
        self.init_consumed_samples = 0

    # doing dataset (custom) setup in the normal set up method ensures that
    # distributed is initialized appropriately by the time the data is loaded,
    # since distributed is needed for NeMo upsampling
    # It is also nice to do the data setup here because the data are not
    # initialized unless this module specifically is trained
    def setup(self, *args, **kwargs):
        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path

        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0

        self.init_consumed_samples = init_consumed_samples

        super().setup(*args, **kwargs)
        self.custom_setup()

    def custom_setup(self):
        self.data_module = SpliceSiteDataModule(
            self.cfg, self.trainer, self.encoder_model
        )
        self._build_train_valid_datasets()
        self.setup_training_data(self.cfg)
        self.setup_validation_data(self.cfg)
        # TODO should we include test data?

    @staticmethod
    def get_hiddens_for_idx(input_tensor, idx):
        return input_tensor[:, idx, :]

    def encoder_forward(self, bert_model, batch: dict):
        tokens, types, _, _, lm_labels, padding_mask = \
            bert_model.process_batch(batch)
        if not self.cfg.bert_binary_head:
            types = None
        output_tensor = bert_model(tokens, padding_mask, token_type_ids=types, lm_labels=lm_labels)
        return output_tensor

    def forward(self, batch: dict):
        output_tensor = self.encoder_forward(self.encoder_model, batch)
        task_input_tensor = self.extract_for_task_head(output_tensor)
        output = self.task_head(task_input_tensor)
        return output

    def _calc_step(self, batch, batch_idx):
        output_tensor = self.forward(batch)
        loss = self.loss_fn(output_tensor, batch[self.batch_target_name])
        return loss

    def training_step(self, batch, batch_idx):
        # TODO: I think we need a more sophisticated solution for DDP loss
        # TODO: ^ reference NeMo
        loss = self._calc_step(batch, batch_idx)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        self.log('reduced_train_loss', reduced_loss, prog_bar=True)
        # if we wanted to enable gradient accumulation across batches we could
        # do something more sophisticated like megatron BERT:
        # https://github.com/NVIDIA/NeMo/blob/c9811f14fa1e1f990fd29f1aed1ae08e2ff6b014/nemo/collections/nlp/models/language_modeling/megatron_bert_model.py#L132-L154
        # self.log('train_loss', loss, prog_bar=True)
        self.log_stats()
        return loss

    def log_stats(self):
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr)
        self.log('global_step', self.trainer.global_step, prog_bar=True)
        # self.log(
        #     'consumed_samples',
        #     self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
        #     prog_bar=True,
        # )

    # # TODO incorporate this
    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def validation_step(self, batch, batch_idx):
        loss = self._calc_step(batch, batch_idx)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        return reduced_loss

    def validation_epoch_end(self, outputs):
        averaged_loss = torch.stack(outputs).mean()
        self.log('val_loss', averaged_loss, prog_bar=True)

    def _build_train_valid_datasets(self):
        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()

    def setup_training_data(self, cfg):
        consumed_samples = self.compute_consumed_samples(0)
        logging.info(
            f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
        )
        self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)
        self.data_module.adjust_train_dataloader(self, self._train_dl)
        print(len(self._train_dl))

    def setup_validation_data(self, cfg):
        consumed_samples = 0
        logging.info(
            f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
        )
        self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)
        self.data_module.adjust_val_dataloader(self, self._validation_dl)

model = SpliceSiteBERTPredictionModel(cfg.model, trainer)

trainer.fit(model)
