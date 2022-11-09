from abc import abstractmethod
import torch
from nemo.utils import logging
from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from apex.transformer import parallel_state
from bionemo.model.utils import (
    extract_consumed_samples_from_ckpt,
    compute_consumed_samples,
)


class EncoderFineTuning(ModelPT, Exportable):

    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer)
        if trainer.accumulate_grad_batches != 1:
            raise ValueError(
                "Trainer.accumulate_grad_batches currently only supported"
                " for Trainer.accumulate_grad_batches = 1")
        self.cfg = cfg
        self.encoder_model = self.setup_encoder_model(cfg, trainer)
        self.init_consumed_samples = 0
        self.loss_fn = self.build_loss_fn()
        self.task_head = self.build_task_head()
        self.predict_dataset = None

    def list_available_models(self):
        return []

    def on_train_start(self) -> None:
        super().on_train_start()
        self.init_global_step = self.trainer.global_step

    def configure_optimizers(self):
        # TODO do we need to configure a distributed optimizer?, similar to here:
        # https://github.com/NVIDIA/NeMo/blob/c9811f14fa1e1f990fd29f1aed1ae08e2ff6b014/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L349
        return super().configure_optimizers()

    @abstractmethod
    def setup_encoder_model(self, cfg, trainer):
        pass

    def extract_for_task_head(self, input_tensor):
        return input_tensor

    @abstractmethod
    def encoder_forward(self, bert_model, batch: dict):
        pass

    @abstractmethod
    def build_loss_fn(self):
        pass

    @abstractmethod
    def build_task_head(self):
        pass

    @abstractmethod
    def get_target_from_batch(self, batch):
        pass

    # doing dataset (custom) setup in the normal set up method ensures that
    # distributed is initialized appropriately by the time the data is loaded,
    # since distributed is needed for NeMo upsampling
    # It is also nice to do the data setup here because the data are not
    # initialized unless this module specifically is trained
    def setup(self, *args, **kwargs):
        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path

        if resume_checkpoint_path:
            init_consumed_samples = extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0

        self.init_consumed_samples = init_consumed_samples
        self.data_setup()
        super().setup(*args, **kwargs)

    def on_fit_start(self):
        self.setup_training_data(self.cfg)
        self.setup_validation_data(self.cfg)
        self.setup_test_data(self.cfg)

    @abstractmethod
    def data_setup(self):
        pass

    def predict_dataloader(self):
        self.predict_dataset = self.data_module.create_dataset(
            self.cfg.data.predict_file
        )
        return torch.utils.data.DataLoader(
            self.predict_dataset, num_workers=self.cfg.data.num_workers,
            pin_memory=True, shuffle=False,
            batch_size=self.cfg.micro_batch_size,
        )

    def forward(self, batch: dict):
        output_tensor = self.encoder_forward(self.encoder_model, batch)
        task_input_tensor = self.extract_for_task_head(output_tensor)
        output = self.task_head(task_input_tensor)
        return output

    def _calc_step(self, batch, batch_idx):
        output_tensor = self.forward(batch)
        loss = self.loss_fn(output_tensor, self.get_target_from_batch(batch))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calc_step(batch, batch_idx)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        self.log('reduced_train_loss', reduced_loss, prog_bar=True)
        # if we wanted to enable gradient accumulation across batches we could
        # do something more sophisticated like megatron BERT:
        # https://github.com/NVIDIA/NeMo/blob/c9811f14fa1e1f990fd29f1aed1ae08e2ff6b014/nemo/collections/nlp/models/language_modeling/megatron_bert_model.py#L132-L154
        self.log_stats()
        return loss

    def log_stats(self):
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr)
        self.log('global_step', self.trainer.global_step, prog_bar=True)
        self.log(
            'consumed_samples',
            compute_consumed_samples(self, self.trainer.global_step - self.init_global_step + 1),
            prog_bar=True,
        )

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

        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def validation_step(self, batch, batch_idx):
        loss = self._calc_step(batch, batch_idx)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        return reduced_loss

    def validation_epoch_end(self, outputs):
        if len(outputs) > 0:
            averaged_loss = torch.stack(outputs).mean()
        else:
            averaged_loss = torch.nan
        self.log('val_loss', averaged_loss, prog_bar=True)

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = compute_consumed_samples(self, 0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)
            self.data_module.adjust_train_dataloader(self, self._train_dl)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)
            self.data_module.adjust_val_dataloader(self, self._validation_dl)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)
            self.data_module.adjust_test_dataloader(self, self._test_dl)
