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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pathlib
import pickle
from typing import Any, List

from pytorch_lightning import Callback, LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader


def get_global_step(trainer: Trainer, pl_module: LightningModule) -> int:
    """Fetches the global step from the trainer state."""
    return trainer.global_step


def get_learning_rate(trainer: Trainer, pl_module: LightningModule) -> float:
    """Fetches the learning rate from the trainers optimizer state."""
    return trainer.optimizers[0].param_groups[0]["lr"]


def get_biobert_val_loss(trainer: Trainer, pl_module: LightningModule, on_save: bool = False) -> float:
    """Only works for the BioBert model when using a single gpu (or ddp)."""
    result = compute_biobert_loss_singlegpu(pl_module, trainer.datamodule.val_dataloader())
    return result


def compute_biobert_loss_singlegpu(model, dl: DataLoader):
    """Computes the loss for BioBert models on a single GPU.

    This will not function in multi-gpu settings nor with models that do not conform to BioBert.

    Args:
        model (torch.nn.Module): The Biobert model.
        dl (torch.utils.data.DataLoader): The data loader.

    Returns:
        float: The mean loss.

    See Also:
    - :class: BioBertModel
    """
    pass

    n, loss = 0, 0.0
    model.eval()
    # batch = next(iter(dl))
    batch = model.data_step(iter(dl))
    result = model(
        input_ids=batch["text"].cuda(),  # 'tokens' also a valid input for MockGPTDataModule
        attention_mask=batch["attention_mask"].cuda(),
    )
    loss_mask = batch["loss_mask"].cuda()
    # Not guaranteed i guess?
    logits = result["token_logits"]
    target = batch["labels"].cuda()
    loss += F.cross_entropy(logits[loss_mask].float(), target[loss_mask], reduction="sum")
    n += loss_mask.sum()

    mean_loss: float = (loss / n).detach().cpu().numpy().item()
    model.train()
    return mean_loss


getter_function_map = {
    "learning_rate": get_learning_rate,
    "global_step": get_global_step,
    "val_biobert_loss": get_biobert_val_loss,
    # How do we compute a simple validation loss here
}


class StopAndGoException(Exception):  # noqa: D101
    pass


class RaiseAfterMetadataCallback(Callback):
    """A callback that raises a StopAndGoException kills it if the metadata from the MetadataSaveCallback was saved successfully beforehand.

    Use this callback for pytest based Stop and go tests.
    """

    def __init__(self, metadata_path: pathlib.Path):  # noqa: D107
        self.metadata_path = metadata_path

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int):
        """PTL callback that raises a StopAndGoException if metadata exists."""
        pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")
        if os.path.exists(pickle_file_path):
            # Register the signal handler
            raise StopAndGoException("Terminating early, checkpoint exists.")
            # kill job afterwards


class MetadataSaveCallback(Callback):
    """A callback that saves metadata about the current training at the second validation epoch."""

    def __init__(self, metadata_path: pathlib.Path, metadata_keys: List[str]):
        """Initialises callback with path and called information.

        Args:
            metadata_path (pathlib.Path): Path where the metadata will be saved.
            metadata_keys (List[str]): keys for metadata to be checked.
        """
        self.metadata_path = metadata_path
        self.metadata_keys = metadata_keys
        self.pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")
        self.called = False  # indicates if callback was already called

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        """Set up the testing callbacks and removes lingering metadata."""
        super().setup(trainer, pl_module, stage)
        if trainer.is_global_zero and os.path.exists(self.pickle_file_path):
            os.remove(self.pickle_file_path)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Stores requisite metadata at the end of the first non-warmup validation epoch.

        Executes on the second validation epoch -only- due to how warmups are handled. May not work as intended in the
        absence of a warmup.

        Args:
            trainer (Trainer): The Lightning Trainer object.
            pl_module (LightningModule): The LightningModule being trained.

        Notes:
            - If `called` is True and `trainer.is_global_zero` is True, the function saves metadata to compare after resuming with a checkpoint.
            - The metadata is obtained using `getter_function_map` and saved as a pickle file.

        """
        if self.called and trainer.is_global_zero:
            # save metadata to compare to after resuming with checkpoint
            metadata = {}
            for metadata_key in self.metadata_keys:
                metadata_value = getter_function_map[metadata_key](trainer, pl_module)
                metadata[metadata_key] = metadata_value
            # prepare paths for metadata save
            pickle_file_path = self.pickle_file_path
            os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)
            with open(pickle_file_path, "wb") as metadata_file:
                pickle.dump(metadata, metadata_file)
            # check that pickle file was saved correctly
            assert os.path.isfile(pickle_file_path), f"No file found at {pickle_file_path}"
        else:
            # first time this callback is called is before the ModelCheckpoint callback
            # since that one is always executed last. Therefore, we skip the first validation
            # round and only save metadata at the second validation round
            self.called = True


class TestCheckpointIntegrityCallback(Callback):
    """Callback that tests if current metrics match those saved in the associated metadata file.

    This callback expects to be invoked _only_ after resuming a model that used the MetadataSaveCallback. When training begins, it checks the value of each metric and compares to the metadata stored in the metadata pickle file. Any deviances are assumed to be a failure in restoration.
    """

    def __init__(self, metadata_path: pathlib.Path, metadata_keys: List[str]):
        """Initialises callback with path and called information.

        Args:
            metadata_path (pathlib.Path): Path where the metadata will be saved.
            metadata_keys (List[str]): keys for metadata to be checked.
        """
        self.metadata_path = metadata_path
        self.metadata_keys = metadata_keys

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        """Loads associated metadata and compares with current metrics."""
        pickle_file_path = os.path.join(self.metadata_path, "checkpoints/metadata.pkl")
        # check that pickle file exists
        assert os.path.isfile(pickle_file_path), f"No file found at {pickle_file_path}"
        with open(pickle_file_path, "rb") as metadata_file:
            metadata_dict = pickle.load(metadata_file)
        current_metadata = {}
        for metadata_key in self.metadata_keys:
            expected_value = metadata_dict[metadata_key]
            current_value = getter_function_map[metadata_key](trainer, pl_module)
            current_metadata[metadata_key] = current_value

        # TODO (SKH): Ideally this would collect _all_ failures instead of failing on the first one.
        for metadata_key in self.metadata_keys:
            expected_value = metadata_dict[metadata_key]
            current_value = current_metadata[metadata_key]
            assert (
                expected_value == current_value
            ), f"Value mismatch for key {metadata_key}: stored_value={expected_value}, current_value={current_value}"
        # Cleanup
        os.remove(pickle_file_path)
