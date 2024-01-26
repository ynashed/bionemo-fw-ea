# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import Logger
from pathlib import Path
from typing import Callable, Generic, List, Optional, Sequence, Tuple, Union

import model_navigator
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from model_navigator.package.package import Package
from model_navigator.runtime_analyzer.strategy import RuntimeSearchStrategy
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf import DictConfig, OmegaConf
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from bionemo.model.core.infer import M
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.triton.types_constants import (
    EMBEDDINGS,
    HIDDENS,
    MASK,
    SEQUENCES,
    NamedArrays,
    SeqsOrBatch,
    StrInferFn,
)
from bionemo.utils.tests import register_searchpath_config_plugin


__all__: Sequence[str] = (
    # binding inference functions to Triton
    "register_str_embedding_infer_fn",
    "register_masked_decode_infer_fn",
    # loading a model's configuration object
    "load_model_config",
    # loading a base encode-decode model for inference, w/ or w/o model navigator support
    "load_model_for_inference",
    "load_nav_package_for_model",
    "load_navigated_model_for_inference",
    "model_navigator_filepath",
    # encoding & decoding strings as numpy arrays
    "decode_str_batch",
    "encode_str_batch",
)


def register_str_embedding_infer_fn(
    triton: Triton,
    infer_fn: StrInferFn,
    triton_model_name: str,
    *,
    output_masks: bool,
    in_name: str = SEQUENCES,
    out: str = EMBEDDINGS,
    out_dtype: np.dtype = np.float32,
    out_shape: Tuple[int] = (-1,),
    max_batch_size: int = 10,
    verbose: bool = True,
) -> None:
    """Binds a string-to-embedding batch inference function to Triton.

    The inference function's input is assumed to be a batch of strings, encoded as utf-8 bytes in a dense numpy array.
    The output will be a single named dense numpy array of floats, corresponding 1:1 to the batch indicies.

    The `in_name` and `out` names correspond to these input and output tensors, respectively. The output embeddings
    may be customized on their underlying datatype and shape. The input, however, is fixed.
    """
    if verbose:
        logging.info(
            f"Binding new inference function in Triton under '{triton_model_name}'\n"
            f"Inference fn. docs: {infer_fn.__doc__} ({type(infer_fn)=})\n"
            f"Input, batched, utf8 text, named:  {in_name}\n"
            f"Output, batched, f32 embeddings:   {out}"
        )
    if any((len(x) == 0 for x in [out, in_name, triton_model_name])):
        raise ValueError(f"Need non-empty values, not {in_name=}, {out=}, {triton_model_name=}")
    if max_batch_size < 1:
        raise ValueError(f"Need positive {max_batch_size=}")

    outputs = [
        Tensor(name=out, dtype=out_dtype, shape=out_shape),
    ]
    if output_masks:
        outputs.append(Tensor(name=MASK, dtype=np.bool_, shape=out_shape))

    triton.bind(
        model_name=triton_model_name,
        infer_func=infer_fn,
        inputs=[Tensor(name=in_name, dtype=bytes, shape=(1,))],
        outputs=outputs,
        config=ModelConfig(max_batch_size=max_batch_size),
    )


def register_masked_decode_infer_fn(
    triton: Triton,
    infer_fn: Callable[[NamedArrays], NamedArrays],
    triton_model_name: str,
    *,
    in_name: str = HIDDENS,
    in_dtype: np.dtype = np.float32,
    in_shape: Tuple[int] = (-1,),
    out: str = SEQUENCES,
    max_batch_size: int = 10,
    verbose: bool = True,
) -> None:
    """Binds a Triton inference function that decodes embeddings, with an input mask, to a batch of strings.

    The inference function's input is a dense batch of embeddings and a mask. While the mask's tensor name is fixed,
    the embeddings input tensor name can be customized via the `in_name` parameter. Additionally, the datatype for
    this tensor is provided by `in_dtype`. The `in_shape` applies to both input tensors: the mask and the embedding.
    Note that the "mask" is a binary array: its datatype is `bool`.

    The expected output of the inference function is a batch of strings, encoded as utf-8 bytes in a dense numpy array.
    The `out` parameter customizes the name of this output tensor.
    """
    if verbose:
        logging.info(
            f"Binding new inference function in Triton under '{triton_model_name}'\n"
            f"Inference fn. docs: {infer_fn.__doc__} ({type(infer_fn)=})\n"
            f"Input, batched, f32 embedding, named:  {in_name}\n"
            f"Output, batched, utf8 text, named:     {out}"
        )
    if any((len(x) == 0 for x in [out, in_name, triton_model_name])):
        raise ValueError(f"Need non-empty values, not {in_name=}, {out=}, {triton_model_name=}")
    if max_batch_size < 1:
        raise ValueError(f"Need positive {max_batch_size=}")

    triton.bind(
        model_name=triton_model_name,
        infer_func=infer_fn,
        inputs=[
            Tensor(name=in_name, dtype=in_dtype, shape=in_shape),
            Tensor(name=MASK, dtype=np.bool_, shape=(-1,)),
        ],
        outputs=[
            Tensor(name=out, dtype=bytes, shape=(1,)),
        ],
        config=ModelConfig(max_batch_size=max_batch_size),
    )


def load_model_config(
    config_path: Union[Path, str], config_name: str, *, logger: Optional[Logger] = None
) -> DictConfig:
    class C(SearchPathPlugin):
        def __init__(self) -> None:
            super().__init__()

        def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
            search_path.prepend(provider="bionemo-searchpath-plugin", path="file:///workspace/bionemo/examples/conf")

    register_searchpath_config_plugin(C)

    with initialize_config_dir(version_base=None, config_dir=str(config_path)):
        cfg = compose(config_name=config_name.replace('.yaml', ''))
    if logger is not None:
        logger.info("\n\n************** Experiment configuration ***********")
        logger.info(f'\n{OmegaConf.to_yaml(cfg)}')
    return cfg


def load_model_for_inference(cfg: DictConfig, *, interactive: bool = False) -> M:
    """Loads a bionemo encoder-decoder model from a complete configuration, preparing it only for inference."""
    if not hasattr(cfg, "infer_target"):
        raise ValueError(f"Expecting configuration to have an `infer_target` attribute. Invalid config: {cfg=}")
    infer_class = import_class_by_path(cfg.infer_target)

    initialize_distributed_parallel_state(
        local_rank=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_split_rank=0,
        interactive=interactive,
    )

    model = infer_class(cfg, interactive=interactive)
    model.freeze()

    return model


def load_navigated_model_for_inference(cfg: DictConfig, strategy: RuntimeSearchStrategy) -> Tuple[M, Package]:
    """Like `load_model_for_inference`, but also uses `load_nav_package_for_model` to load model's optimized runtime."""
    nav_path = model_navigator_filepath(cfg)

    model = load_model_for_inference(cfg)

    runner = load_nav_package_for_model(model, nav_path, strategy)
    runner.activate()

    return model, runner


def model_navigator_filepath(cfg: DictConfig) -> str:
    """Convention for obtaining the Model Navigator artifact path from a large config object."""
    if hasattr(cfg, 'nav_path'):
        return cfg.nav_path
    else:
        return f"{cfg.model.downstream_task.restore_from_path[: -len('.nemo')]}.nav"


class NavWrapper(torch.nn.Module, Generic[M]):
    def __init__(self, inferer: M) -> None:
        """WARNING: side effects: moves the inferer to the cuda device and calls its _prepare_for_export method."""
        super().__init__()
        self.model = inferer.model.cuda()
        self.prepare_for_export()

    def prepare_for_export(self) -> None:
        self.model._prepare_for_export()

    def forward(self, tokens_enc: torch.Tensor, enc_mask: torch.Tensor):
        """Alias for the model's encoder forward-pass."""
        return self.model.encode(tokens_enc=tokens_enc, enc_mask=enc_mask)


def load_nav_package_for_model(
    model: M,
    nav_path: str,
    strategy: RuntimeSearchStrategy,
) -> Package:
    """Loads the previously exported model navigator optimized runtime for the given model."""
    nav_model = NavWrapper(model)

    package = model_navigator.package.load(nav_path)
    package.load_source_model(nav_model)

    runner = package.get_runner(strategy=strategy)

    return runner


def decode_str_batch(sequences: np.ndarray) -> List[str]:
    """Decodes a utf8 byte array into a batch of string sequences."""
    seqs = np.char.decode(sequences.astype("bytes"), "utf-8")
    seqs = seqs.squeeze(1)
    return seqs.tolist()


def encode_str_batch(sequences: SeqsOrBatch) -> np.ndarray:
    """Encodes a batch of string sequences as a dense utf8 byte array."""
    if len(sequences) == 0:
        raise ValueError("Need at least one sequence to encode")
    if isinstance(sequences[0], str):
        # List[str] case
        seqs: List[List[str]] = [[s] for s in sequences]
    else:
        # assume List[List[str]] case
        seqs = sequences
    return np.char.encode(np.array(seqs), encoding='utf-8')
