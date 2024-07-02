# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from bionemo.model.molecule.moco.data.adaptive_dataloader import AdaptiveBatchSampler
from bionemo.model.molecule.moco.data.midi_dataloader import MiDiDataloader
from bionemo.model.molecule.moco.data.molecule_dataset import MoleculeDataset


class MoleculeDataModule(LightningDataModule):
    """
    PyTorch Lightning Data Module for molecular data.

    Args:
        dataset_root (str): Root directory of the dataset.
        processed_folder (str): Folder containing processed data.
        batch_size (int, optional): Batch size for training DataLoader.
        removed_h (bool, optional): Whether to remove hydrogen atoms. Defaults to False.
        data_loader_type (str, optional): Type of DataLoader ('adaptive', 'standard', 'dynamic', or 'midi'). Defaults to 'adaptive'.
        inference_batch_size (int, optional): Batch size for validation, test, and predict DataLoader. Defaults to batch_size.
        **sampler_kwargs: Additional keyword arguments for the sampler.
    """

    def __init__(
        self,
        dataset_root: str,
        processed_folder: str,
        batch_size: Optional[int] = None,
        removed_h: bool = False,
        data_loader_type: str = "adaptive",
        inference_batch_size: Optional[int] = None,
        **sampler_kwargs,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.processed_folder = processed_folder
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size or batch_size
        self.removed_h = removed_h
        self.data_loader_type = data_loader_type
        self.sampler_kwargs = sampler_kwargs
        self.pin_memory = True

        self.train_dataset = MoleculeDataset(
            split="train",
            root=self.dataset_root,
            processed_folder=self.processed_folder,
            removed_h=self.removed_h,
            only_stats=False,
        )
        self.val_dataset = MoleculeDataset(
            split="val",
            root=self.dataset_root,
            processed_folder=self.processed_folder,
            removed_h=self.removed_h,
            only_stats=False,
        )
        self.test_dataset = MoleculeDataset(
            split="test",
            root=self.dataset_root,
            processed_folder=self.processed_folder,
            removed_h=self.removed_h,
            only_stats=False,
        )

        self.statistics = {
            "train": self.train_dataset.statistics,
            "val": self.val_dataset.statistics,
            "test": self.test_dataset.statistics,
        }

    def train_dataloader(self):
        """Returns the DataLoader for training dataset."""
        return self._create_dataloader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Returns the DataLoader for validation dataset."""
        return self._create_dataloader(self.val_dataset, self.inference_batch_size, shuffle=False)

    def test_dataloader(self):
        """Returns the DataLoader for test dataset."""
        return self._create_dataloader(self.test_dataset, self.inference_batch_size, shuffle=False)

    def predict_dataloader(self):
        """Returns the DataLoader for prediction dataset."""
        return self._create_dataloader(self.test_dataset, self.inference_batch_size, shuffle=False)

    def _create_dataloader(self, dataset, batch_size, shuffle):
        """Creates a DataLoader for a given dataset.

        Args:
            dataset (Dataset): The dataset for which to create the DataLoader.
            batch_size (int): Batch size for DataLoader.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: The DataLoader for the given dataset.
        """
        if self.data_loader_type == "adaptive":
            sampler = AdaptiveBatchSampler(dataset, reference_batch_size=batch_size, **self.sampler_kwargs)
        elif self.data_loader_type == "dynamic":
            sampler = DynamicBatchSampler(dataset, **self.sampler_kwargs)
        elif self.data_loader_type == "midi":
            return MiDiDataloader(dataset, batch_size=batch_size, shuffle=shuffle, **self.sampler_kwargs)
        else:
            sampler = None

        if sampler is not None:
            return DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=self.pin_memory)
        else:
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=self.pin_memory
            )


def test_datamodule():
    """Function to test the MoleculeDataModule with adaptive, standard, dynamic, and midi samplers."""
    dataset_root = "/data/pyg_geom_drug"
    processed_folder = "processed_tiny"
    batch_size = 128
    removed_h = False

    # Test with adaptive sampler
    adaptive_datamodule = MoleculeDataModule(
        dataset_root=dataset_root,
        processed_folder=processed_folder,
        batch_size=batch_size,
        inference_batch_size=batch_size,
        removed_h=removed_h,
        data_loader_type="adaptive",
        max_num_nodes=None,
        lookahead_size=800,
    )

    print("Testing with Adaptive Sampler:")
    test_dataloader = adaptive_datamodule.test_dataloader()
    total_size = 0
    for batch in test_dataloader:
        assert hasattr(batch, "mol")
        total_size += len(batch.mol)
    assert len(adaptive_datamodule.test_dataset) == total_size

    # Test with standard sampler
    standard_datamodule = MoleculeDataModule(
        dataset_root=dataset_root,
        processed_folder=processed_folder,
        batch_size=batch_size,
        removed_h=removed_h,
        data_loader_type="standard",
    )

    print("Testing with Standard Sampler:")
    test_dataloader = standard_datamodule.test_dataloader()
    total_size = 0
    for batch in test_dataloader:
        assert hasattr(batch, "mol")
        total_size += len(batch.mol)
    assert len(standard_datamodule.test_dataset) == total_size

    # Test with dynamic sampler
    dynamic_datamodule = MoleculeDataModule(
        dataset_root=dataset_root,
        processed_folder=processed_folder,
        batch_size=batch_size,
        removed_h=removed_h,
        data_loader_type="dynamic",
        max_num=5000,  # Example parameter for edge or node-based sampling
        mode="node",  # Change to "edge" if edge-based sampling is preferred
    )

    print("Testing with Dynamic Sampler:")
    test_dataloader = dynamic_datamodule.test_dataloader()
    total_size = 0
    for batch in test_dataloader:
        assert hasattr(batch, "mol")
        total_size += len(batch.mol)
    assert len(dynamic_datamodule.test_dataset) == total_size

    # Test with MiDiDataloader
    midi_datamodule = MoleculeDataModule(
        dataset_root=dataset_root,
        processed_folder=processed_folder,
        batch_size=batch_size,
        inference_batch_size=batch_size,
        removed_h=removed_h,
        data_loader_type="midi",
    )

    print("Testing with MiDiDataloader:")
    test_dataloader = midi_datamodule.test_dataloader()
    total_size = 0
    for batch in test_dataloader:
        assert hasattr(batch, "mol")
        total_size += len(batch.mol)
    print(f"Loader yielded {total_size} graphs, with dataset size {len(midi_datamodule.test_dataset)}")


if __name__ == "__main__":
    test_datamodule()
