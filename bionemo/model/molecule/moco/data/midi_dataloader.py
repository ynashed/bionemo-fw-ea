# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Mapping, Sequence
from typing import List, Optional, Union

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData

from bionemo.model.molecule.moco.data.adaptive_dataloader import effective_batch_size


class AdaptiveCollater:
    def __init__(self, follow_batch, exclude_keys, reference_batch_size, reference_size):
        """Copypaste from pyg.loader.Collater + small changes"""
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.reference_bs = reference_batch_size
        self.reference_size = reference_size

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            to_keep = []
            graph_sizes = []

            for e in batch:
                e: BaseData
                graph_sizes.append(e.num_nodes)

            m = len(graph_sizes)
            graph_sizes = torch.Tensor(graph_sizes)
            srted, argsort = torch.sort(graph_sizes)
            # random = m // 2 # torch.randint(0, m, size=(1, 1)).item()
            random = torch.randint(0, m, size=(1, 1)).item()
            max_size = min(srted.max().item(), srted[random].item() + 5)
            max_size = max(max_size, 9)  # The batch sizes may be huge if the graphs happen to be tiny

            ebs = effective_batch_size(max_size, self.reference_bs, reference_size=self.reference_size)

            max_index = torch.nonzero(srted <= max_size).max().item()
            min_index = max(0, max_index - ebs)
            indices_to_keep = set(argsort[min_index : max_index + 1].tolist())
            if max_index < ebs:
                for index in range(max_index + 1, m):
                    # Check if we could add the graph to the list
                    size = srted[index].item()
                    potential_ebs = effective_batch_size(size, self.reference_bs, reference_size=self.reference_size)
                    if len(indices_to_keep) < potential_ebs:
                        indices_to_keep.add(argsort[index].item())

            for i, e in enumerate(batch):
                e: BaseData
                if i in indices_to_keep:
                    to_keep.append(e)

            new_batch = Batch.from_data_list(to_keep, self.follow_batch, self.exclude_keys)
            return new_batch

        elif True:
            # early exit
            raise NotImplementedError("Only supporting BaseData for now")
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")

    def collate(self, batch):  # Deprecated...
        return self(batch)


class MiDiDataloader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` into mini-batches, each minibatch being a bucket with num_nodes < some threshold,
    except the last which holds the overflow-graphs. Apart from the bucketing, identical to torch_geometric.loader.DataLoader
    Default bucket_thresholds is [30,50,90], yielding 4 buckets
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        reference_batch_size: int = 1,
        reference_size: int = 20,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size=reference_batch_size,
            shuffle=shuffle,
            collate_fn=AdaptiveCollater(
                follow_batch,
                exclude_keys,
                reference_batch_size=reference_batch_size,
                reference_size=reference_size,
            ),
            **kwargs,
        )
