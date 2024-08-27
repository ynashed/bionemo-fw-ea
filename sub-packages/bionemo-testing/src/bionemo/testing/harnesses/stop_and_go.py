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


import pathlib
from abc import ABC, abstractmethod
from typing import Literal

import nemo.lightning as nl
import pytorch_lightning as pl
from nemo.collections import llm
from nemo.lightning import nemo_logger, resume
from nemo.lightning.pytorch import callbacks as nl_callbacks

from bionemo.testing import testing_callbacks
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state


class StopAndGoHarness(ABC):
    """Abstract base class for a stop-and-go harness. Stop and go tests act as follows:
        - setup a clean model for a brief training run, select metrics to track.
        - interrupt training via the StopAndGoException in the callback InterruptAfterMetadataCallback.
        - setup a model to be resumed from the checkpoint, with the same metrics.
        - Restore training and check that metadta matches the stored metrics in the callback CheckpointIntegrityCallback.
      Useful metrics to check are things like learning rate, global step, validation loss, training loss, and anything
        else that is important to the training process. If there is an unavailable metrics, a method for fetching the
        metric should be provided in the bionemo.testing.callbacks module.

    Considerations when implementing this class:
        - devices, pipeline_model_parallel, and tensor_model_parallel may impact the setup of DataModule. Certain
            datasets expect a known global batch size, which depends on the number of devices and conditional
            tensor model parallel/ pipeline model parallel settings.
        - 'mode' is useful in some cases, but not in all cases. Implement conditions based on these when useful. As an
            example, it may be useful to implement a test that stops and resumes with different parallelism settings.
            - changing callbacks to test metadata integrity (core feature of stop-and-go tests).
            - changing trainer behavior to use multiple GPUs
            - changing the model construction to use different hyperparameters.
            - ... etc
            Each of the above tests cases may be useful for automated testing of various expected behavior.
        - stop(), go(), and run_test() are provided methods which execute the actual tests, leveraging the conditions
            in the various setup methods, respecting 'mode' where necessary.

    Attributes:
        root_dir (str): The root directory.
        exp_name (str): The experiment name.
        metadata_dir (str): The metadata directory.
        metrics (list[str]): The list of metrics.
        nemo_logger (NeMoLogger): The NeMo logger.
        val_check_interval (int): The validation check interval. Stored as an attribute to ensure consistency between
            validation

    Methods:
        __init__(self, metrics: list[str], exp_name='stop_and_go_harness'): Initializes the StopAndGoHarness object.
        setup_model(self, mode: Literal['stop', 'go']) -> tuple[BioBertLightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]: Constructs the model, data, and optimizer for the test harness.
        get_callbacks(self, mode: Literal['stop', 'go'], metrics): Returns the callbacks based on the mode.
        setup_trainer_and_strategy(self, mode: Literal["stop", "go"], metrics): Sets up the trainer and strategy.
        stop(self): Runs the requisite methods with the 'stop' mode.
        go(self): Runs the requisite methods with the 'go' mode.
        run_test(self): Executes the stop => go process.

    See Also: bionemo.testing.callbacks.
    """

    def __init__(
        self,
        metrics: list[str],
        root_dir: pathlib.Path | str = pathlib.Path("./"),
        val_check_interval=2,
        exp_name="stop_and_go_harness",
    ):
        self.root_dir = root_dir  # Set to bionemo2_home ideally.
        self.exp_name = exp_name
        self.metadata_dir = self.root_dir / self.exp_name
        self.metrics = metrics
        self.val_check_interval = val_check_interval
        self.nemo_logger: nemo_logger.NeMoLogger = nemo_logger.NeMoLogger(
            dir=str(self.root_dir),
            name=self.exp_name,
            use_datetime_version=False,
            version=None,
            tensorboard=None,
            wandb=None,
            ckpt=None,
        )

    @abstractmethod
    def setup_model(
        self, mode: Literal["stop", "go"]
    ) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        """Constructs the model, data, and optimizer for the test harness. Optionally supports separate code paths for 'stop'/'go', although implementors are
        encouraged to use the same code path for both.
        """
        ...

    @abstractmethod
    def setup_trainer_and_strategy(self, mode: Literal["stop", "go"], metrics) -> pl.Trainer:
        """Constructs the trainer object for the stop and go test. This method invokes the
        get_callbacks method to get the appropriate callbacks for the mode and passes it to the trainer.
        """
        ...

    def get_callbacks(self, mode: Literal["stop", "go"], metrics: list[str]) -> list[pl.Callback]:
        """Returns a list of callbacks based on the specified mode. Base implemention provides reasonable defaults.

        To extend this method, call the super and append to the callbacks, depending on which mode you are in:

        ```python
        callbacks = super().get_callbacks(mode, metrics)
        callbacks.append(MyCustomCallback())
        return callbacks
        ```

        Args:
            mode (Literal['stop', 'go']): The mode indicating whether to stop or go.
            metrics: The metrics to be used for tracking.

        Returns:
            list: A list of callbacks based on the specified mode.

        Raises:
            ValueError: If the mode is neither 'stop' nor 'go'.
        """
        if mode == "stop":
            callbacks = [
                testing_callbacks.MetadataSaveCallback(
                    metadata_path=self.metadata_dir,
                    metadata_keys=metrics,
                ),
                testing_callbacks.RaiseAfterMetadataCallback(metadata_path=self.metadata_dir),
                nl_callbacks.ModelCheckpoint(
                    save_best_model=False,
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=self.val_check_interval,
                    enable_nemo_ckpt_io=True,
                    try_restore_best_ckpt=False,
                ),
            ]
        elif mode == "go":
            # we must setup the integrity callback.
            callbacks = [
                testing_callbacks.TestCheckpointIntegrityCallback(
                    metadata_path=self.metadata_dir, metadata_keys=metrics
                ),
                nl_callbacks.ModelCheckpoint(
                    save_best_model=False,
                    save_last=True,
                    monitor="reduced_train_loss",
                    save_top_k=2,
                    every_n_train_steps=self.val_check_interval,
                    enable_nemo_ckpt_io=True,
                    try_restore_best_ckpt=False,
                ),
            ]
        else:
            raise ValueError("mode must be 'stop' or 'go'")

        return callbacks

    # stop() and go() are provided methods and run the requisite methods with the appropriate mode.
    def stop(self):
        model, data, opt = self.setup_model(mode="stop")
        trainer = self.setup_trainer_and_strategy("stop", self.metrics)
        with distributed_model_parallel_state():
            try:
                llm.train(
                    model=model,
                    data=data,
                    trainer=trainer,
                    log=self.nemo_logger,
                    optim=opt,
                    resume=resume.AutoResume(
                        path=None,  # Overrides the path found by resume_if_exists when set.
                        resume_if_exists=False,  # Looks for the -last checkpoint to continue training.
                        resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                    ),
                )
            except testing_callbacks.StopAndGoException:
                ...

    def go(self):
        model, data, opt = self.setup_model(mode="go")
        trainer = self.setup_trainer_and_strategy("go", self.metrics)
        with distributed_model_parallel_state():
            llm.train(
                model=model,
                data=data,
                trainer=trainer,
                log=self.nemo_logger,
                optim=opt,
                resume=resume.AutoResume(
                    path=None,  # Overrides the path found by resume_if_exists when set.
                    resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                ),
            )

    # Finally, execution is a simple stop => go.
    def run_test(self):
        self.stop()
        self.go()
