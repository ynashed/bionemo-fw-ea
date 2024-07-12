# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional

import pytorch_lightning as L
import torch
import torch.distributed
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.lightning import get_vocab_size, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from torch.optim import Optimizer


if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

    from bionemo.contrib.model.biobert import MegatronBioBertModel


@dataclass
class GeneformerConfig(TransformerConfig):
    # From megatron.core.models.gpt.bert_model.GPTModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024

    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False

    optimizer_fn: Optional[Callable[["MegatronBioBertModel"], Optimizer]] = None

    def configure_model(self, tokenizer) -> "MegatronBioBertModel":
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state
        from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec

        from bionemo.contrib.model.biobert import MegatronBioBertModel

        do_next_sentence = False
        return MegatronBioBertModel(
            self,
            transformer_layer_spec=bert_layer_with_transformer_engine_spec,
            num_tokentypes=2 if do_next_sentence else 0,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            return_embeddings=False,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),  # set to False for inference
            add_binary_head=do_next_sentence,
        )


class GeneformerModel(L.LightningModule, io.IOMixin, io.ConnectorMixin):
    def __init__(
        self,
        config: GeneformerConfig,
        # TODO: Add transformer_layer_spec when we update mcore
        tokenizer: Optional["TokenizerSpec"] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def configure_model(self) -> None:
        self.module = self.config.configure_model(self.tokenizer)

    def configure_optimizers(self) -> Optimizer:
        if self.config.optimizer_fn is not None:
            return self.config.optimizer_fn(self)

        return bert_default_optimizer(self)

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        output_tensor = self.module(*args, **kwargs)  # for now just pass through to the underlying model
        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return geneformer_data_step(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return bert_forward_step(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        # This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss
        #  This function will
        return MaskedTokenLossReduction()

    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        return MaskedTokenLossReduction(validation_step=True)

    def copy(self) -> "GeneformerModel":
        return self.__class__(self.config, self.tokenizer)


def geneformer_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.add("text")
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask", "types", "is_random"))
    # if self.get_attention_mask_from_fusion:
    #     required_keys.remove('attention_mask')

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def bert_forward_step(model, batch) -> torch.Tensor:
    """
    input_ids: Tensor,
    attention_mask: Tensor,
    tokentype_ids: Tensor = None,
    lm_labels: Tensor = None,
    inference_params=None,
    """
    forward_args = {
        "input_ids": batch["text"],
        "attention_mask": batch["attention_mask"],
        # "tokentype_ids": batch.get("types", None),  # TODO support tokentypes when they are meaningful.
        "lm_labels": batch["labels"],
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    logits, binary_head_logits = model(**forward_args)
    return logits  # TODO support losses that also include the binary head, this means doing something more fancy than the one default GPT reduction function above MaskedTokenLossReduction()


def bert_default_optimizer(module) -> Optimizer:
    from apex.optimizers import FusedAdam

    return FusedAdam(module.parameters(), lr=1e-4)


def get_batch_on_this_context_parallel_rank(batch):
    from megatron.core import parallel_state

    if cp_size := parallel_state.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if 'loss_mask' in batch and batch['loss_mask'] is not None:
            num_valid_tokens_in_ub = batch['loss_mask'].sum()

        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2
                _val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                    non_blocking=True
                )
                _val = _val.index_select(seq_dim, index)
                _val = _val.view(*val.shape[0:seq_dim], -1, *_val.shape[(seq_dim + 2) :])
                batch[key] = _val
        batch['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub
    return batch


def get_packed_seq_params(batch):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = batch['cu_seqlens'].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if cu_seqlens_argmin := batch.get('cu_seqlens_argmin', None) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch['max_seqlen'].squeeze() if 'max_seqlen' in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )


__all__ = [
    "GeneformerModel",
    "GeneformerConfig",
    "geneformer_data_step",
    "bert_forward_step",
    "bert_default_optimizer",
]
