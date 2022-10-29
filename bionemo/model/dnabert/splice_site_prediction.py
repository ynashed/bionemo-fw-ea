from torch import nn
from bionemo.model.dnabert import DNABERTModel
from bionemo.data.dna.splice_site_dataset import SpliceSiteDataModule
from bionemo.model.core.encoder_finetuning import EncoderFineTuning
from bionemo.model.core import MLPModel

class SpliceSiteBERTPredictionModel(EncoderFineTuning):

    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer=trainer)

        # START: Unique Setup
        # TODO make number_of_classes (3) configurable for classificaiton
        # TODO make other NN MLP parameters configurable
        # TODO make MLPModel configurable
        self.task_head = MLPModel(layer_sizes=[cfg.hidden_size, 3], dropout=0.1)

        # TODO make the loss configurable
        self.loss_fn = nn.CrossEntropyLoss() #define a loss function
        # TODO make target name configurable from cfg?
        self.batch_target_name = 'target'
        # TODO double check that this index is the correct one (according to get mid point function)
        # TODO and make it based off of the sequence length
        self.extract_idx = 200

    def modify_encoder_model(self, encoder_model):
        encoder_model.model.post_process = False

    def setup_encoder_model(self, cfg, trainer):
        # TODO this could be refactored to instantiate a new model if no
        # checkpoint is specified

        model = DNABERTModel.load_from_checkpoint(
            # TODO grab ckpt from config
            cfg.encoder_checkpoint,
            cfg=cfg,
            trainer=trainer,
        )
        self.modify_encoder_model(model)

        return model

    def extract_for_task_head(self, input_tensor):
        return self.get_hiddens_for_idx(input_tensor, idx=self.extract_idx)

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

    def data_setup(self):
        self.data_module = SpliceSiteDataModule(
            self.cfg, self.trainer, self.encoder_model
        )
        self.build_train_valid_test_datasets()

    def build_train_valid_test_datasets(self):
        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()
