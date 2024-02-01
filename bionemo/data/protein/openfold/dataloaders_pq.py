import queue
import threading
from copy import deepcopy
from multiprocessing.managers import SyncManager
from typing import Iterator, List

import torch

from bionemo.data.protein.openfold.dataloaders import TrainBatchProperties
from bionemo.data.protein.openfold.datasets import InitialTrainingDataset
from bionemo.data.protein.openfold.helpers import collate
from bionemo.data.protein.openfold.samplers import InitialTrainingSampler


class InitialTrainingDataloaderPQ:
    """Dataloader for the initial training stage with non-blocking priority queue."""

    def __init__(
        self,
        dataset: InitialTrainingDataset,
        sampler: InitialTrainingSampler,
        local_batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        seed: int,
        uniform_recycling_iters: List[int],
        num_prev_iters: int,
        use_threading: bool,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.local_batch_size = local_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._set_train_batch_properties_fn = TrainBatchProperties(
            seed=seed,
            uniform_recycling_iters=uniform_recycling_iters,
            num_prev_iters=num_prev_iters,
        )
        self.queue_maxsize = self.num_workers * self.prefetch_factor
        if torch.distributed.is_initialized():
            self.dist_rank = int(torch.distributed.get_rank())
            self.dist_world_size = int(torch.distributed.get_world_size())
        else:
            self.dist_rank = None
            self.dist_world_size = None

        # [optim-hub] since DAP is not implemented, dist_group_size, dist_rank_in_group, and
        # dist_producer_rank will always be None. If DAP continues to be out of scope, we can optimise the
        # code and remove these arguments

        self.dist_group_size = None
        self.dist_rank_in_group = None
        self.dist_group = None
        self.threading_enabled = use_threading

    def _start_manager(self) -> None:
        self._manager = ManagerPQ()
        self._manager.start()

    def _start_multiprocessing(self) -> None:
        # create queues:
        self._index_queue = torch.multiprocessing.Queue(maxsize=self.queue_maxsize)
        self._sample_pqueue = self._manager.PriorityQueue(maxsize=self.queue_maxsize)
        self._batch_pqueue = self._manager.PriorityQueue(maxsize=self.queue_maxsize)

        # create index process:
        self._index_process = torch.multiprocessing.Process(
            target=_index_worker,
            args=(
                self.sampler,
                self._index_queue,
            ),
        )
        self._index_process.daemon = True
        self._index_process.start()

        # create sample processes:
        self._sample_processes = []
        for w in range(self.num_workers):
            sample_process = torch.multiprocessing.Process(
                target=_sample_worker,
                args=(
                    self.dataset,
                    self._index_queue,
                    self._sample_pqueue,
                ),
            )
            sample_process.daemon = True
            sample_process.start()
            self._sample_processes.append(sample_process)

        # create batch process:
        self._batch_process = torch.multiprocessing.Process(
            target=_batch_worker,
            args=(
                self._sample_pqueue,
                self._batch_pqueue,
                self.local_batch_size,
            ),
        )
        self._batch_process.daemon = True
        self._batch_process.start()

    def _start_threading(self, input_queue) -> None:
        if self.threading_enabled:
            self._batch_tqueue = queue.Queue(maxsize=1)
            self._batch_thread = threading.Thread(
                target=_batch_thread,
                args=(
                    input_queue,
                    self._batch_tqueue,
                ),
            )
            self._batch_thread.daemon = True
            self._batch_thread.start()

    def _close_manager(self) -> None:
        if hasattr(self, "_manager"):
            del self._manager

    def _close_multiprocessing(self) -> None:
        if hasattr(self, "_batch_process"):
            self._batch_process.terminate()
            del self._batch_process

        if hasattr(self, "_sample_processes"):
            for sample_process in reversed(self._sample_processes):
                sample_process.terminate()
            del self._sample_processes

        if hasattr(self, "_index_process"):
            self._index_process.terminate()
            del self._index_process

        if hasattr(self, "_batch_pqueue"):
            del self._batch_pqueue

        if hasattr(self, "_sample_pqueue"):
            del self._sample_pqueue

        if hasattr(self, "_index_queue"):
            del self._index_queue

    def _close_threading(self) -> None:
        if hasattr(self, "_batch_thread"):
            del self._batch_thread

        if hasattr(self, "_batch_tqueue"):
            del self._batch_tqueue

    def _multiprocessing_iter(self) -> Iterator[dict]:
        self._start_manager()
        self._start_multiprocessing()
        self._start_threading(self._batch_pqueue)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            if self.threading_enabled:
                priority, batch = self._batch_tqueue.get(timeout=2)
            else:
                priority, batch = self._batch_pqueue.get(timeout=2)
            batch["__priority__"] = priority
            yield batch
        self._close_threading()
        self._close_multiprocessing()
        self._close_manager()

    def _synchronous_iter(self) -> Iterator[dict]:
        sampler_iterator = iter(self.sampler)
        sampler_length = len(self.sampler)
        num_dataloader_iters = sampler_length // self.local_batch_size
        for i in range(num_dataloader_iters):
            samples = []
            for j in range(self.local_batch_size):
                index = next(sampler_iterator)
                sample = self.dataset[index]
                assert isinstance(sample, dict)
                samples.append(sample)
            batch = collate(samples)
            batch["__priority__"] = i
            yield batch

    def __iter__(self) -> Iterator[dict]:
        if self.num_workers > 0:
            iterator = self._multiprocessing_iter()
        elif self.num_workers == 0:
            iterator = self._synchronous_iter()
        for i, batch in enumerate(iterator):
            yield self._set_train_batch_properties_fn(batch)

    def __del__(self) -> None:
        self._close_threading()
        self._close_multiprocessing()
        self._close_manager()


def _index_worker(sampler, index_queue) -> None:
    for priority, index in enumerate(sampler):
        index_queue.put((priority, index))


def _sample_worker(dataset, index_queue, sample_pqueue) -> None:
    while True:
        priority, index = index_queue.get()
        sample = dataset[index]
        sample_pqueue.put((priority, sample))


def _batch_worker(sample_pqueue, batch_pqueue, local_batch_size: int) -> None:
    while True:
        samples = []
        priorities = []
        for j in range(local_batch_size):
            priority, sample = sample_pqueue.get()
            samples.append(sample)
            priorities.append(priority)
        batch = collate(samples)
        priority = min(priorities)
        batch_pqueue.put((priority, batch))


def _batch_thread(input_queue, batch_tqueue) -> None:
    while True:
        obj = input_queue.get()
        batch_tqueue.put(obj)


def _batch_deepcopy(batch: dict) -> dict:
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.clone()
        else:
            output[key] = deepcopy(value)
    return output


class ManagerPQ(SyncManager):
    pass


ManagerPQ.register("PriorityQueue", queue.PriorityQueue)
