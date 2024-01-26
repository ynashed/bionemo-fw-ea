# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional, Union

import torch
from diffdock import TPCUDA as TensorProduct
from diffdock.TensorProduct import FullyConnectedTPFunction
from nemo.utils import logging
from torch.nn.modules.module import T


class FullyConnectedTP(torch.nn.Module):
    def __init__(
        self,
        in_irreps: str,
        sh_irreps: str,
        out_irreps: str,
        dtype: torch.dtype = torch.float32,
        cuda_device: Optional[Union[int, torch.device]] = 0,
    ):
        super().__init__()
        self._in_irreps = in_irreps
        self._out_irreps = out_irreps
        self._sh_irreps = sh_irreps
        self._device = torch.device(cuda_device)
        self._dtype = dtype
        if self._device.type != 'cuda':
            raise RuntimeError(f'Only cuda device is supported for FullyConnectedTP, but got device = {self._device}')
        self._tp: TensorProduct = None
        self._init_tp()
        self.weight_numel = self._tp.weight_numel

    def forward(self, u, v, weight):
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return self._tp.forward(u.to(self._dtype), v.to(self._dtype), weight.to(self._dtype))
        else:
            return FullyConnectedTPFunction.apply(
                u.to(self._dtype), v.to(self._dtype), weight.to(self._dtype), self._tp
            )

    def _init_tp(self: T) -> T:
        """initialize self._tp with given device and dtype.

        Returns:
            Module: self
        """
        if self._device == torch.device('cuda'):
            self._device = torch.device('cuda:0')
        self._tp = TensorProduct(
            str(self._in_irreps),
            str(self._sh_irreps),
            str(self._out_irreps),
            dtype=self._dtype,
            device=str(self._device),
        )
        return self

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self
        """

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        update_device = False
        update_dtype = False
        if dtype is not None:
            if not dtype.is_floating_point or dtype != torch.float32:
                raise TypeError(
                    'nn.Module.to only accepts floating point dtype, but got desired dtype={}'.format(dtype)
                )
            if self._dtype != dtype:
                self._dtype = dtype
                update_dtype = True

        if device is not None:
            if device == torch.device('cpu'):
                logging.warning("FullyConnectedTP can only be used with cuda device, as such move to device('cuda:0')")
            if device == torch.device('cuda') or device == torch.device('cpu'):
                device = torch.device('cuda:0')
            if self._device != device:
                self._device = device
                update_device = True

        if update_dtype or update_device:
            return self._init_tp()
        else:
            return self

    def _apply(self, fn):
        return fn(self)

    def is_floating_point(self):
        return self._dtype.is_floating_point

    def float(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        self.dtype = torch.float
        return self._init_tp()

    def double(self: T) -> T:
        raise RuntimeError('FullyConnectedTP does not support double precision yet')

    def half(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        raise RuntimeError("FullyConnectedTP does not support half precision yet")

    def bfloat16(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        raise RuntimeError("FullyConnectedTP does not support bfloat16 yet")

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        if device is None:
            return self
        else:
            self._device = torch.device(device)
            return self._init_tp()

    def cpu(self: T) -> T:
        r"""Moves all model parameters and buffers to the CPU.

        .. note::
            FullyConnectedTP can only be used with cuda device, as such .cpu() is doing nothing here.

        Returns:
            Module: self
        """
        logging.warning("FullyConnectedTP can only be used with cuda device, as such .cpu() is doing nothing here")
        return self
