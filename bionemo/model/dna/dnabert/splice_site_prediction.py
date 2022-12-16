from functools import lru_cache
from omegaconf import OmegaConf
from torch import nn
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from bionemo.model.dna.dnabert import DNABERTModel
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

        # TODO do we need to override any keys in the encoder_cfg?
        # e.g., tensor_model_parallel_size and pipeline_model_parallel_size

        model = DNABERTModel.restore_from(
            restore_path=cfg.encoder.checkpoint,
            trainer=trainer,
            save_restore_connector=NLPSaveRestoreConnector(),
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

    # the lru cache is kind of a hacky way to make sure this isn't set up if
    # it is already initialized, since this function doesn't return anything
    @lru_cache
    def data_setup(self):
        self.data_module = SpliceSiteDataModule(
            self.cfg, self.trainer, self.encoder_model,
        )

    def on_fit_start(self):
        self.build_train_valid_test_datasets()
        return super().on_fit_start()

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
