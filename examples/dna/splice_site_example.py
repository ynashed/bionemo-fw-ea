from bionemo.model.utils import setup_trainer
from bionemo.model.dnabert.splice_site_prediction import SpliceSiteBERTPredictionModel
from omegaconf import OmegaConf


base_cfg_file = '/workspace/bionemo/examples/dna/conf/dnabert_base_config.yaml'
cfg_file = '/workspace/bionemo/examples/dna/conf/dnabert_config_splice_site.yaml'

base_cfg = OmegaConf.load(base_cfg_file)
cfg = OmegaConf.load(cfg_file)
cfg = OmegaConf.merge(base_cfg, cfg)

# cfg.trainer['strategy'] = None
# cfg.exp_manager.wandb_logger_kwargs.notes = ''
# cfg.exp_manager.wandb_logger_kwargs.offline = True
# cfg.model.tokenizer.k = 3
ckpt = '/workspace/bionemo/examples/dna/nemo_experiments/dnabert/2022-10-25_00-04-34/checkpoints/dnabert--val_loss=8.57-step=760400-consumed_samples=3041600.0-last.ckpt'
cfg.trainer.max_steps = 5000 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches

gff_csv = '/workspace/bionemo/examples/dna/data/splice-site-prediction/sampled-data-10k/sampled-data.csv'
cfg.model.data['train_file'] = gff_csv
cfg.model.data['val_file'] = gff_csv
cfg.model.data['fasta_directory'] = '/workspace/bionemo/examples/dna/data/splice-site-prediction/GRCh38.ensembl.99'
cfg.model.data['fasta_pattern'] = 'Homo_sapiens.GRCh38.dna.chromosome.{}.fa.gz'
cfg.model['encoder_checkpoint'] = ckpt
cfg.model.num_workers = 1

trainer = setup_trainer(cfg)
model = SpliceSiteBERTPredictionModel(cfg.model, trainer)
trainer.fit(model)
