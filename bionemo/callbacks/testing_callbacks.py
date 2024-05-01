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
import pickle
import signal

from nemo.utils import logging
from pytorch_lightning import Callback


class KillAfterSignalCallback(Callback):
    """A callback that sends a SIGTERM signal to the process and kills it if the metadata
    from the MetadataSaveCallback was saved successfully beforehand.
    """

    def __init__(self, metadata_path):
        self.metadata_path = metadata_path

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        def terminate_process(signum, frame):
            logging.info("\nReceived SIGTERM signal, terminating process...\n")
            exit(0)

        pickle_file_path = os.path.join(self.metadata_path, 'checkpoints/metadata.pkl')
        # check if metadata has been saved by MetadataSaveCallback
        if os.path.exists(pickle_file_path):
            # Register the signal handler
            signal.signal(signal.SIGTERM, terminate_process)
            # kill job afterwards
            os.kill(os.getpid(), signal.SIGTERM)


class MetadataSaveCallback(Callback):
    """A callback that saves metadata about the current training at the second validation epoch."""

    def __init__(self, metadata_path):
        """Initialises callback with path and called information.
        Args:
            metadata_path (pathlib.Path): Path where the metadata will be saved.
        """
        self.metadata_path = metadata_path
        self.called = False  # indicates if callback was already called

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.called:
            # save metadata to compare to after resuming with checkpoint
            metadata = {'global_step': trainer.global_step, 'lr': trainer.optimizers[0].param_groups[0]["lr"]}
            # prepare paths for metadata save
            pickle_file_path = os.path.join(self.metadata_path, 'checkpoints/metadata.pkl')
            os.makedirs(os.path.dirname(pickle_file_path), exist_ok=True)
            with open(pickle_file_path, 'wb') as metadata_file:
                pickle.dump(metadata, metadata_file)
            # check that pickle file was saved correctly
            assert os.path.isfile(pickle_file_path), f"No file found at {pickle_file_path}"

        else:
            # first time this callback is called is before the ModelCheckpoint callback
            # since that one is always executed last. Therefore, we skip the first validation
            # round and only save metadata at the second validation round
            self.called = True


class TestCheckpointIntegrityCallback(Callback):
    """A callback that, after resuming from a checkpoint, checks that attributes of this
    resumed checkpoint are the same as the metadata that was saved at the time of
    checkpoint creation as part of stop-and-go tests.
    """

    def __init__(self, metadata_path):
        self.metadata_path = metadata_path

    def on_train_start(self, trainer, pl_module):
        pickle_file_path = os.path.join(self.metadata_path, 'checkpoints/metadata.pkl')
        # check that pickle file exists
        assert os.path.isfile(pickle_file_path), f"No file found at {pickle_file_path}"
        with open(pickle_file_path, 'rb') as metadata_file:
            metadata_dict = pickle.load(metadata_file)
        assert (
            trainer.global_step == metadata_dict["global_step"]
        ), f"Global step mismatch for trainer: {trainer.global_step} != {metadata_dict['global_step']}"
        assert (
            trainer.optimizers[0].param_groups[0]["lr"] == metadata_dict["lr"]
        ), f"Learning rate mismatch for trainer: {trainer.optimizers[0].param_groups[0]['lr']} != {metadata_dict['lr']}"
