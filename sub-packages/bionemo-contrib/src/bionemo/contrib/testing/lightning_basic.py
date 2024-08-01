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

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from megatron.core import ModelParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction, ReductionT
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST


@dataclass
class ExampleConfig(ModelParallelConfig):
    """Timers from ModelParallelConfig are required (apparently).
    For megatron forward compatibility.

    """  # noqa: D205

    calculate_per_token_loss: bool = False

    def configure_model(self) -> torch.nn.Module:
        """This function is called by the strategy to construct the model.

        Note: Must pass self into Model since model requires having a config object.

        Returns:
            The model object.
        """
        return ExampleModel(self)


class MSELossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: DataT, forward_out: torch.Tensor) -> Tuple[torch.Tensor, ReductionT]:
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
        x_hat = forward_out
        xview = x.view(x.size(0), -1)
        loss = nn.functional.mse_loss(x_hat, xview)

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> torch.Tensor:
        """Works across micro-batches. (data on single gpu).
        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """  # noqa: D205
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


class LitAutoEncoder(pl.LightningModule):
    """A very basic lightning module example that is used for testing the megatron strategy and the megatron-nemo2-bionemo
    contract.
    """  # noqa: D205

    def __init__(self, config):
        """Args:
        config: a Config object necessary to construct the actual torch.nn.Module (the thing that has the parameters).
        """  # noqa: D205
        super().__init__()
        self.config = config
        self.optim = MegatronOptimizerModule(
            config=OptimizerConfig(lr=1e-4, optimizer="adam", use_distributed_optimizer=True),
        )
        # Bind the configure_optimizers method to the model
        self.optim.connect(self)

    def forward(self, batch: Dict, batch_idx: int):
        """This forward will be called by the megatron scheduler and it will be wrapped.
        Note: The `training_step` defines the training loop and is independent of the `forward` method here.

        Args:
            batch: A dictionary of data.
            batch_idx: The index of the batch.

        Returns:
            The output of the model.
        """  # noqa: D205
        x = batch["data"]
        return self.module(x)

    def training_step(self, batch, batch_idx=None):  # noqa: D417
        """Background:
        - NeMo's Strategy overrides this method.
        - The strategies' training step will call the forward method of the model.
        - That forward method then calls the wrapped forward step of MegatronParallel which wraps the forward method of the model.
        - That wrapped forward step is then executed inside the Mcore scheduler, which calls the `_forward_step` method from the
            MegatronParallel class.
        - Which then calls the training_step function here.

        In this particular use case, we simply call the forward method of this class, the lightning module.

        Args:
            batch: A dictionary of data.
            requires `batch_idx` as default None.
        """  # noqa: D205
        return self(batch, batch_idx)

    def training_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        # This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss
        return MSELossReduction()

    def validation_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        return MSELossReduction()

    def test_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        return MSELossReduction()

    def configure_model(self) -> None:  # noqa: D102
        self.module = self.config.configure_model()


class ExampleModel(MegatronModule):  # noqa: D101
    def __init__(self, config: ModelParallelConfig):
        """Constructor of the model.

        Args:
            config: An instance of a ModelParallelConfig. The config object is responsible
                for telling the strategy what model to create.
        """
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.linear1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 3)
        self.linear3 = nn.Linear(3, 64)
        self.relu2 = nn.ReLU()
        self.linear4 = nn.Linear(64, 28 * 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: torch.Tensor containing the input data.

        Returns:
            x_hat: the result of the last linear layer of the network.
        """
        x = x.view(x.size(0), -1)
        z = self.linear1(x)
        z = self.relu(z)
        z = self.linear2(z)
        x_hat = self.linear3(z)
        x_hat = self.relu2(x_hat)
        x_hat = self.linear4(x_hat)
        return x_hat

    def set_input_tensor(self, input_tensor: Optional[torch.Tensor]) -> None:
        """This is needed because it is a megatron convention. Even if it is a no-op for single GPU testing.

        See megatron.model.transformer.set_input_tensor()

        Note: Currently this is a no-op just to get by an mcore function.

        Args:
            input_tensor: Input tensor.
        """
        pass


class MNISTCustom(MNIST):  # noqa: D101
    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, int]]:
        """Wraps the getitem method of the MNIST dataset such that we return a Dict
        instead of a Tuple or tensor.

        Args:
            index: The index we want to grab, an int.

        Returns:
            Dict containing the items we want for training.
                x: data
                y: label
                idx: index of the data
        """  # noqa: D205
        x, y = super().__getitem__(index)

        return {
            "data": x,
            "label": y,
            "idx": index,
        }


class MNISTDataModule(pl.LightningDataModule):  # noqa: D101
    def __init__(self, data_dir: str = "./", batch_size: int = 32):  # noqa: D107
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.micro_batch_size = 8
        self.global_batch_size = 8
        self.max_len = 100
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

    def setup(self, stage: str):
        """Sets up the datasets

        Args:
            stage: can be one of train / test / predict.
        """  # noqa: D415
        self.mnist_test = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=False)
        self.mnist_predict = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=False)
        mnist_full = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=True)
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):  # noqa: D102
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):  # noqa: D102
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):  # noqa: D102
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self):  # noqa: D102
        return torch.utils.data.DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=0)
