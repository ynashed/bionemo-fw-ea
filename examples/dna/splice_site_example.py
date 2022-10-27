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


from nemo.collections.nlp.parts.nlp_overrides import (
    NLPSaveRestoreConnector,
)
from bionemo.model.utils import setup_trainer, TrainerBuilder
from bionemo.model.dnabert import DNABERTModel
from omegaconf import OmegaConf
from nemo.core.config import hydra_runner
import pytorch_lightning as pl
from torch.utils.data import default_collate
from nemo.utils.app_state import AppState
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel

import torch
import os
from nemo.utils import logging

def initialize_distributed(local_rank, backend='nccl'):
    """Initialize torch.distributed."""
    # Get local rank in case it is provided.
    # local_rank = args.local_rank

    # Get rank and world size.
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    logging.info(
        f'Initializing torch.distributed with local_rank: {local_rank}, rank: {rank}, world_size: {world_size}'
    )

    # Set the device id.
    device = rank % torch.cuda.device_count()
    if local_rank is not None:
        device = local_rank
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank, init_method=init_method)
    return local_rank, rank, world_size

def initialize_distributed_alt(trainer, reconfigure_microbatch=True):
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
    )
    from apex.transformer import parallel_state
    if parallel_state.is_unitialized():

        def dummy():
            return

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(dummy, trainer=trainer)
        trainer.strategy.setup_environment()

        # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
        if reconfigure_microbatch:
            _reconfigure_microbatch_calculator(
                rank=0,  # This doesn't matter since it is only used for logging
                rampup_batch_size=None,
                global_batch_size=1,
                micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
                data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
            )


def save_model(state_dict, f):
    import h5py
    f = h5py.File(f, 'w')
    for k, v in state_dict.items():
        f[k] = v.detach().cpu().numpy()

class DNABERTSaveRestoreConnector(NLPSaveRestoreConnector):
    # TODO: find the way to get rid of 128 constant
    # 128 -- is the number of padded vocabulary in MegatronT5Model
    def __init__(self) -> None:
        super().__init__()

    def modify_state_dict(self, conf, state_dict):
        # save_model(state_dict, 'restore-state-dict.hdf5')
        # print(state_dict)
        return state_dict


base_cfg_file = '/workspace/bionemo/examples/dna/conf/dnabert_base_config.yaml'
cfg_file = '/workspace/bionemo/examples/dna/conf/dnabert_config_splice_site.yaml'

base_cfg = OmegaConf.load(base_cfg_file)
cfg = OmegaConf.load(cfg_file)
cfg = OmegaConf.merge(base_cfg, cfg)



class CustomTrainer(TrainerBuilder):
    pass
    # @staticmethod
    # def configure_plugins(cfg):
    #     plugins = TrainerBuilder.configure_plugins(cfg)
    #     return plugins[1:]
    @staticmethod
    def resume_checkpoint(cfg, trainer):
        pass

    @staticmethod
    def init_model_parallel(cfg, trainer):
        pass
        # initialize_distributed(0)
        # app_state = AppState()
        # # app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
        # app_state.model_parallel_size = 1
        # (
        #     app_state.tensor_model_parallel_rank,
        #     app_state.pipeline_model_parallel_rank,
        #     app_state.model_parallel_size,
        #     app_state.data_parallel_size,
        #     app_state.pipeline_model_parallel_split_rank,
        # ) = fake_initialize_model_parallel(
        #     # world_size=app_state.model_parallel_size,
        #     world_size=1,
        #     rank=trainer.global_rank,
        #     # tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
        #     tensor_model_parallel_size_=1,
        #     # pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
        #     pipeline_model_parallel_size_=1,
        #     # pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
        #     pipeline_model_parallel_split_rank_=1,
        # )


cfg.trainer['strategy'] = None
# cfg.trainer.precision = 32
cfg.exp_manager.wandb_logger_kwargs.notes = ''
cfg.exp_manager.wandb_logger_kwargs.offline = True
cfg.model.tokenizer.k = 3
# nemo_file = 'example-checkpoints/overfit-converted.nemo'
nemo_file = 'example-checkpoints/new-overfit-model.nemo'
ckpt = '/workspace/bionemo/examples/dna/nemo_experiments/dnabert/2022-10-25_00-04-34/checkpoints/dnabert--val_loss=8.57-step=760400-consumed_samples=3041600.0-last.ckpt'
# cfg.model.resume_from_checkpoint = ckpt
cfg.trainer.max_steps = 5000000 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
trainer = setup_trainer(cfg, CustomTrainer())

app_state = AppState()

# model = DNABERTModel.restore_from(
#     nemo_file,
#     cfg.model,
#     trainer=trainer,
#     save_restore_connector=DNABERTSaveRestoreConnector(),
# )

# model = DNABERTModel(cfg.model, trainer)
# save_model(model.state_dict(), 'new-model.hdf5')

gff_csv = '/workspace/bionemo/examples/dna/data/splice-site-prediction/sampled-data-10k/sampled-data.csv'
cfg.model.data['train_file'] = gff_csv
cfg.model.data['val_file'] = gff_csv
cfg.model.data['fasta_directory'] = '/workspace/bionemo/examples/dna/data/splice-site-prediction/GRCh38.ensembl.99'
cfg.model.data['fasta_pattern'] = 'Homo_sapiens.GRCh38.dna.chromosome.{}.fa.gz'
cfg.model.num_workers = 1

model = DNABERTModel.load_from_checkpoint(
    ckpt,
    cfg=cfg.model,
    trainer=trainer,
)
# model.half()
# model.freeze()
# model = DNABERTModel(cfg.model, trainer)
from torch.utils.data import DataLoader
opt = model.optimizers()

from bionemo.data.utils import MappedDataset

from splice_site_data_module import SpliceSiteDataModule

class MMAPSlice(MappedDataset):

    def create_sample_mapping(self, dataset, num_samples):
        import numpy as np
        mmap = np.memmap('/workspace/bionemo/examples/dna/data/small-example/train/chr1-trim-train.fna_train_indexmap_20000000mns_510msl_0.00ssp_1235s.npy')
        mapping = mmap[:4]
        return mapping

ds = MMAPSlice(model.data_module.train_dataset())
# dl = DataLoader(ds, batch_size=cfg.model.micro_batch_size, shuffle=False)
# model.data_module.adjust_train_dataloader(model, dl)
splice_site_dm = SpliceSiteDataModule(cfg.model, trainer, model)
model.data_module = splice_site_dm
initialize_distributed_alt(trainer)
model.setup()

from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable

from bionemo.model.core import MLPModel
from torch import nn

class SpliceSitePredictionModel(ModelPT, Exportable):
    pass

    def __init__(self, cfg, trainer):
        #self._check_scheduler(cfg)
        super().__init__(cfg, trainer=trainer)
        self.cfg = cfg

        # TODO make this load intead
        self.pretrained_model = model

        # TODO make 3 configurable for classificaiton
        # TODO make MLPModel configurable
        self.downstream_model = MLPModel(layer_sizes=[512, 3], dropout=0.1)
        # TODO make (get_embedding_from_model_for_mlp) configurable
        self.loss_fn = nn.CrossEntropyLoss() #define a loss function

        # TODO initialize inside this model
        self.data_module = splice_site_dm

        self._build_train_valid_datasets()
        self.setup_training_data(self.cfg)
        self.setup_validation_data(self.cfg)




# from bionemo.data.dataloader.kmer_collate import KmerBertCollate
# from bionemo.utils.metrics import masked_k_correct_total

# for batch in dl:
#     pd.DataFrame(batch['tokens'].T).to_csv('training_batch', sep='\t', index=False)
#     outputs = model.batch_forward(batch)[0].float()
#     argmin = outputs.argmax(dim=2)
#     pd.DataFrame(argmin.T.cpu()).to_csv('training_pred', sep='\t', index=False)
#     pd.DataFrame(batch['labels'].T).to_csv('training_label', sep='\t', index=False)
#     x = batch['tokens']
#     y = batch['labels']

#     print(masked_k_correct_total(x, y, outputs.cpu(), model.tokenizer.get_mask_id(), k=3))
#     print(masked_k_correct_total(x, y, outputs.cpu(), k=128))

trainer.fit(model)


# exit()


# model.save_to('example-checkpoints/overfit-converted.nemo')

# trainer.fit(model)
