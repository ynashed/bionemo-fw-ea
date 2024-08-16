# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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

"""This is intended to be a minimal self-container NeMo2 example."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Type, TypedDict

import pytorch_lightning as pl
import torch
from megatron.core import ModelParallelConfig, dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from nemo.lightning import io
from nemo.lightning.megatron_parallel import MegatronLossReduction, ReductionT
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from bionemo.core.data.resamplers import PRNGDatasetShuffler
from bionemo.llm.lightning import LightningPassthroughPredictionMixin
from bionemo.llm.model.config import MegatronBioNeMoTrainableModelConfig


__all__: Sequence[str] = (
    "ExampleConfig",
    "MSELossReduction",
    "LitAutoEncoder",
    "ExampleModel",
    "MNISTCustom",
    "MNISTDataModule",
)

#############################################################################################
# Core utility functions: Below are some utility functions that allow for loading a nemo2
#  trained model back into a newly initialized megatron core model. The key insight is that
#  the nemo2 lightning module owns a single `self.module = config.configure_model(...)`
#  object. This `config.configure_module(...)` object is the megatron model that we want
#  to load weights into. So we need to adjust the checkpoint keys since they will all
#  have the extra `module.` prefix on them, while the megatron model we just initialized
#  will not.


def munge_key_megatron_to_nemo2(k: str) -> str:
    return f"module.{k}"


def munge_sharded_tensor_key_megatron_to_nemo2(v: ShardedTensor) -> ShardedTensor:
    # This works with PP=1, how do we handle PP>1?
    key = v.key
    v.key = munge_key_megatron_to_nemo2(key)
    return v


def key_in_filter(k: str, filter: Set[str]) -> bool:
    for prefix in filter:
        if k.startswith(prefix):
            return True
    return False


def load_weights_sharded_inplace_nemo2_to_mcore(model, ckpt, skip_loading_keys: Set[str]):
    # Loads checkpoint from state dict
    sharded_state_dict = {
        munge_key_megatron_to_nemo2(k): munge_sharded_tensor_key_megatron_to_nemo2(v)
        for k, v in model.sharded_state_dict().items()
        if not key_in_filter(k, skip_loading_keys)
    }
    dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=ckpt,
        strict=dist_checkpointing.serialization.StrictHandling.ASSUME_OK_UNEXPECTED,
    )


#############################################################################################
# Losses: here we define some loss functions. The output of forward happens in parallel
#  and that is where backward happens. def reduce is only used for collecting forward output
#  for inference, as well as for logging.


class MSELossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: "MnistItem", forward_out: Dict[str, Tensor]) -> Tuple[Tensor, ReductionT]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        x = batch["data"]
        x_hat = forward_out["x_hat"]
        xview = x.view(x.size(0), -1).to(x_hat.dtype)
        loss = nn.functional.mse_loss(x_hat, xview)

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


class MSEPlusClassifierLossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: "MnistItem", forward_out: Dict[str, Tensor]) -> Tuple[Tensor, ReductionT]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        x = batch["data"]
        digits = batch["label"]
        x_hat = forward_out["x_hat"]
        digit_logits = forward_out["digit_logits"]
        xview = x.view(x.size(0), -1).to(x_hat.dtype)
        mse_loss = nn.functional.mse_loss(x_hat, xview)
        classifier_loss = nn.functional.cross_entropy(digit_logits, digits)
        loss = classifier_loss + mse_loss
        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


class ClassifierLossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: "MnistItem", forward_out: Tensor) -> Tuple[Tensor, ReductionT]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        digits = batch["label"]
        digit_logits = forward_out
        loss = nn.functional.cross_entropy(digit_logits, digits)
        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


#########################################################
# Models: These need to be megatron modules. At the most basic level this just means:
#  1. they need a config argument of type ModelParallelConfig
#  2. they need a self.model_type:ModelType enum defined (ModelType.encoder_or_decoder is probably usually fine)
#  3. def set_input_tensor(self, input_tensor) needs to be present. This is used in model parallelism
# TODO add example where we specify things like shape etc to the model so it's clear how we recommend how to do this.


class ExampleModelTrunk(MegatronModule):
    def __init__(self, config: ModelParallelConfig) -> None:
        """Constructor of the model.

        Args:
            config: The config object is responsible for telling the strategy what model to create.
        """
        super().__init__(config)
        # FIXME add an assertion that the user is not trying to do tensor parallelism since this doesn't use
        #  parallelizable megatron linear layers.
        self.model_type: ModelType = ModelType.encoder_or_decoder
        self.linear1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> Tensor:
        # we could return a dictionary of strings to tensors here, but let's demonstrate this is not necessary
        x = x.view(x.size(0), -1)
        z = self.linear1(x)
        z = self.relu(z)
        z = self.linear2(z)
        return z

    def set_input_tensor(self, input_tensor: Optional[Tensor]) -> None:
        """This _would_ be needed for model parallel and other kinds of more complicated forward passes in megatron."""
        pass


class ExampleModel(ExampleModelTrunk):  # noqa: D101
    def __init__(self, config: ModelParallelConfig) -> None:
        """Constructor of the model.

        Args:
            config: The config object is responsible for telling the strategy what model to create.
        """
        super().__init__(config)
        self.linear3 = nn.Linear(3, 64)
        self.relu2 = nn.ReLU()
        self.linear4 = nn.Linear(64, 28 * 28)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass of the model.

        Args:
            x: The input data.

        Returns:
            x_hat: The result of the last linear layer of the network.
        """
        z: Tensor = super().forward(x)
        x_hat = self.linear3(z)
        x_hat = self.relu2(x_hat)
        x_hat = self.linear4(x_hat)
        return {"x_hat": x_hat, "z": z}


class ExampleFineTuneBothModel(ExampleModel):
    """Example of taking the example model and adding an output task."""

    def __init__(self, config: ModelParallelConfig):
        super().__init__(config)
        # 10 output digits, and use the latent output layer (z) for making predictions
        self.digit_classifier = nn.Linear(self.linear2.out_features, 10)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        parent_out: Dict[str, Tensor] = super().forward(x)
        digit_logits = self.digit_classifier(parent_out["z"])
        return {**parent_out, "digit_logits": digit_logits}


class ExampleFineTuneDropParentModel(ExampleModelTrunk):
    """Example of taking the example model and replacing output task."""

    def __init__(self, config: ModelParallelConfig):
        super().__init__(config)
        # 10 output digits, and use the latent output layer (z) for making predictions
        self.digit_classifier = nn.Linear(self.linear2.out_features, 10)

    def forward(self, x: Tensor) -> Tensor:
        z: Tensor = super().forward(x)
        digit_logits = self.digit_classifier(z)  # to demonstrate flexibility, in this case we return a tensor
        return digit_logits


#################################################################################################################
# Model+Loss Configs: these have a configure_model function which allows the megatron strategy to lazily initialize
#  the model after the parallel computing environment has been setup. These also handle loading starting weights
#  for fine-tuning cases. Additionally these configs tell the trainer which loss you want to use with a matched
#  model.


@dataclass
class ExampleConfig(MegatronBioNeMoTrainableModelConfig["ExampleModel", "MSELossReduction"]):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    initial_weights: str | None = None
    calculate_per_token_loss: bool = False
    skip_weight_prefixes: Set[str] = field(default_factory=set)

    def configure_model(self) -> "ExampleModel":
        """This function is called by the strategy to construct the model.

        Note: Must pass self into Model since model requires having a config object.

        Returns:
            The model object.
        """
        # TODO: if checkpoint is provided, load config settings from checkpoint before we intialize the model.
        # cckpt_settings = ...
        # self.n_layers = cckpt_settings.n_layers # something like that
        model = ExampleModel(self)
        if self.initial_weights:
            load_weights_sharded_inplace_nemo2_to_mcore(model, self.initial_weights, self.skip_weight_prefixes)
        return model

    def get_loss_reduction_class(self) -> Type[MSELossReduction]:
        """Return the expected loss reduction class."""
        return MSELossReduction


@dataclass
class ExampleFineTuneBothConfig(
    MegatronBioNeMoTrainableModelConfig["ExampleFineTuneBothModel", "MSEPlusClassifierLossReduction"]
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    initial_weights: Path | None = None
    calculate_per_token_loss: bool = False
    skip_weight_prefixes: Set[str] = field(default_factory=set)

    def configure_model(self) -> "ExampleFineTuneBothModel":
        """This function is called by the strategy to construct the model.

        Note: Must pass self into Model since model requires having a config object.

        Returns:
            The model object.
        """
        model = ExampleFineTuneBothModel(self)
        if self.initial_weights:
            load_weights_sharded_inplace_nemo2_to_mcore(model, self.initial_weights, self.skip_weight_prefixes)
        return model

    def get_loss_reduction_class(self) -> Type["MSEPlusClassifierLossReduction"]:
        return MSEPlusClassifierLossReduction


@dataclass
class ExampleFineTuneDropParentConfig(
    MegatronBioNeMoTrainableModelConfig["ExampleFineTuneDropParentModel", "ClassifierLossReduction"]
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    initial_weights: Path | None = None
    calculate_per_token_loss: bool = False
    skip_weight_prefixes: Set[str] = field(default_factory=set)

    def configure_model(self) -> "ExampleFineTuneDropParentModel":
        """This function is called by the strategy to construct the model.

        Note: Must pass self into Model since model requires having a config object.

        Returns:
            The model object.
        """
        model = ExampleFineTuneDropParentModel(self)
        if self.initial_weights:
            load_weights_sharded_inplace_nemo2_to_mcore(model, self.initial_weights, self.skip_weight_prefixes)
        return model

    def get_loss_reduction_class(self) -> Type["ClassifierLossReduction"]:
        return ClassifierLossReduction


################################################################################
# General training wrapper that can be re-used for all model/loss combos
#  just specify different configs. TODO make this an ABC since it will likely
#  not change much between models, other than the data step and forward step.


class LitAutoEncoder(pl.LightningModule, io.IOMixin, LightningPassthroughPredictionMixin):
    """A very basic lightning module for testing the megatron strategy and the megatron-nemo2-bionemo contract."""

    def __init__(self, config: MegatronBioNeMoTrainableModelConfig):
        """Initializes the model.

        Args:
            config: a Config object necessary to construct the actual nn.Module (the thing that has the parameters).
        """
        super().__init__()
        self.config = config
        self.optim = MegatronOptimizerModule(
            config=OptimizerConfig(lr=1e-4, optimizer="adam", use_distributed_optimizer=True),
        )
        # Bind the configure_optimizers method to the model
        self.optim.connect(self)

    def forward(self, batch: Dict, batch_idx: int) -> Any:
        """This forward will be called by the megatron scheduler and it will be wrapped.

        !!! note

            The `training_step` defines the training loop and is independent of the `forward` method here.

        Args:
            batch: A dictionary of data.
            batch_idx: The index of the batch.

        Returns:
            The output of the model.
        """
        x = batch["data"]
        return self.module(x)

    def training_step(self, batch, batch_idx: Optional[int] = None):
        """The training step is where the loss is calculated and the backpropagation is done.

        Background:
        - NeMo's Strategy overrides this method.
        - The strategies' training step will call the forward method of the model.
        - That forward method then calls the wrapped forward step of MegatronParallel which wraps the forward method of the model.
        - That wrapped forward step is then executed inside the Mcore scheduler, which calls the `_forward_step` method from the
            MegatronParallel class.
        - Which then calls the training_step function here.

        In this particular use case, we simply call the forward method of this class, the lightning module.

        Args:
            batch: A dictionary of data. requires `batch_idx` as default None.
            batch_idx: The index of the batch.
        """
        return self(batch, batch_idx)

    def training_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        # This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss
        return self.loss_reduction_class()()

    def validation_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        return self.loss_reduction_class()()

    def test_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        return self.loss_reduction_class()()

    def configure_model(self) -> None:  # noqa: D102
        # Called lazily by the megatron strategy.
        self.module = self.config.configure_model()

    def loss_reduction_class(self) -> Type[MegatronLossReduction]:
        """Get the loss reduction class the user has specified in their config."""
        return self.config.get_loss_reduction_class()


#######################################################################################
# Data methods. The dataset has no changes vs a vanilla pytorch dataset. The data module
#  has a data_sampler in it which is a nemo2 peculiarity. Also the sampler will not
#  shuffle your data! So you need to wrap your dataset in a dataset shuffler that maps
#  sequential ids to random ids in your dataset.
# TODO make an ABC for nemo2 DataModules
#  which allow us to re-use some of these common functions and not forget to implement
#  the key things that nemo2 uses/needs.
class MnistItem(TypedDict):
    data: Tensor
    label: Tensor
    idx: int


class MNISTCustom(MNIST):  # noqa: D101
    def __getitem__(self, index: int) -> MnistItem:
        """Wraps the getitem method of the MNIST dataset such that we return a Dict
        instead of a Tuple or tensor.

        Args:
            index: The index we want to grab, an int.

        Returns:
            A dict containing the data ("x"), label ("y"), and index ("idx").
        """  # noqa: D205
        x, y = super().__getitem__(index)

        return {
            "data": x,
            "label": y,
            "idx": index,
        }


#######################################################################################
# Data module needs a data_sampler for handling the mcore strategy nemo2 runner.
class MNISTDataModule(pl.LightningDataModule):  # noqa: D101
    def __init__(self, data_dir: str = "./", batch_size: int = 32, global_batch_size: int | None = None) -> None:  # noqa: D107
        super().__init__()
        self.data_dir = data_dir
        self.micro_batch_size = batch_size
        self.global_batch_size = global_batch_size or batch_size
        self.max_len = 1048  # Unused?
        self.rampup_batch_size = None

        #  Note that this sampler is sequential, meaning it does not do any shuffling. Let's wrap our data in a shuffler.
        # Wraps the datasampler with the MegatronDataSampler. The MegatronDataSampler is a wrapper that allows the sampler
        # to be used with megatron. It sets up the capability to utilize micro-batching and gradient accumulation. It is also
        # the place where the global batch size is constructed.
        self.data_sampler = MegatronDataSampler(
            seq_len=self.max_len,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
        )

    def setup(self, stage: str) -> None:
        """Sets up the datasets

        Args:
            stage: can be one of train / test / predict.
        """  # noqa: D415
        self.mnist_test = PRNGDatasetShuffler(
            MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=False), seed=43
        )
        mnist_full = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=True)
        mnist_train, mnist_val = torch.utils.data.random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        self.mnist_train = PRNGDatasetShuffler(mnist_train, seed=44)
        self.mnist_val = PRNGDatasetShuffler(mnist_val, seed=45)

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(self.mnist_train, batch_size=self.micro_batch_size, num_workers=0)

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(self.mnist_val, batch_size=self.micro_batch_size, num_workers=0)

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(self.mnist_test, batch_size=self.micro_batch_size, num_workers=0)