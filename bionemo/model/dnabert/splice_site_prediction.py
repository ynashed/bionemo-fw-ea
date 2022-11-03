from omegaconf import OmegaConf
from torch import nn
from bionemo.model.dnabert import DNABERTModel
from bionemo.data.dna.splice_site_dataset import SpliceSiteDataModule
from bionemo.model.core.encoder_finetuning import EncoderFineTuning
from bionemo.model.core import MLPModel

class SpliceSiteBERTPredictionModel(EncoderFineTuning):

    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer=trainer)

        self.batch_target_name = cfg.target_name
        # use this to get the embedding of the midpoint of the sequence
        self.extract_idx = (cfg.seq_length - 1) // 2

    def build_loss_fn(self):
        return nn.CrossEntropyLoss()

    def build_task_head(self):
        return MLPModel(layer_sizes=[self.cfg.hidden_size, self.cfg.n_outputs],
            dropout=0.1,
        )

    def get_target_from_batch(self, batch):
        return batch[self.batch_target_name]

    def setup_encoder_model(self, cfg, trainer):
        # TODO this could be refactored to instantiate a new model if no
        # checkpoint is specified

        encoder_cfg = OmegaConf.load(cfg.encoder.hparams).cfg
        # TODO do we need to override any keys in the encoder_cfg?
        # e.g., tensor_model_parallel_size and pipeline_model_parallel_size
        model = DNABERTModel.load_from_checkpoint(
            cfg.encoder.checkpoint,
            cfg=encoder_cfg,
            trainer=trainer,
        )
        # TODO should we be doing this with some sort of
        # context management so it can be reversed?
        model.model.post_process = False

        return model

    def extract_for_task_head(self, input_tensor):
        return self.get_hiddens_for_idx(input_tensor, idx=self.extract_idx)

    @staticmethod
    def get_hiddens_for_idx(input_tensor, idx):
        return input_tensor[:, idx, :]

    def encoder_forward(self, bert_model, batch: dict):
        tokens, types, _, _, lm_labels, padding_mask = \
            bert_model.process_batch(batch)
        if not bert_model.cfg.bert_binary_head:
            types = None
        output_tensor = bert_model(tokens, padding_mask, token_type_ids=types, lm_labels=lm_labels)
        return output_tensor

    def data_setup(self):
        self.data_module = SpliceSiteDataModule(
            self.cfg, self.trainer, self.encoder_model
        )
        self.build_train_valid_test_datasets()

    def build_train_valid_test_datasets(self):
        # if we want to make _train_ds optional for testing, we should be able
        # to enforce it with something like an `on_fit_start` method
        self._train_ds = self.data_module.get_sampled_train_dataset()
        val_dataset = self.data_module.get_sampled_val_dataset()
        if len(val_dataset) > 0:
            self._validation_ds = val_dataset
        test_dataset = self.data_module.get_sampled_test_dataset()
        if len(test_dataset) > 0:
            self._test_ds = test_dataset
