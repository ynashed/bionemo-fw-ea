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


from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Type, TypeVar

import torch
from torch import Tensor
from torch.nn.functional import silu
from torch.optim import Optimizer

from bionemo.amplify.data.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.esm2.model.attention import ESM2TEDotProductAttention
from bionemo.esm2.model.embedding import ESM2Embedding
from bionemo.esm2.model.model import ESM2Model

from bionemo.llm.model.biobert.model import BioBertConfig, MegatronBioBertModel, PositionEmbeddingKinds
from bionemo.llm.api import MegatronLossType
from bionemo.llm.utils import iomixin_utils as iom
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption

from megatron.core import tensor_parallel
from megatron.core.models.bert.pooler import Pooler
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer import spec_utils
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import get_linear_layer


__all__: Sequence[str] = (
    "AMPLIFYConfig",
    "AMPLIFYModel",
)

class AMPLIFYModel(ESM2Model):
    """AMPLIFY protein language model."""
    pass

AMPLIFYModelT = TypeVar("AMPLIFYModelT", bound=AMPLIFYModel)

@dataclass
class AMPLIFYConfig(BioBertConfig[AMPLIFYModelT, MegatronLossType], iom.IOMixinWithGettersSetters):
    """Configuration class for AMPLIFY model.

    Attributes:
        num_layers: Number of layers in the model.
        hidden_size: Hidden size of the model.
        num_attention_heads: Number of attention heads in the model.
        ffn_hidden_size: Hidden size of the feed-forward network.
        hidden_dropout: Dropout rate for hidden layers.
        attention_dropout: Dropout rate for attention layers.
        apply_residual_connection_post_layernorm: Whether to apply residual connection after layer normalization.
        layernorm_epsilon: Epsilon value for layer normalization.
        layernorm_zero_centered_gamma: Whether to zero-center the gamma parameter in layer normalization.
        activation_func: Activation function used in the model.
        init_method_std: Standard deviation for weight initialization.
        apply_query_key_layer_scaling: Whether to apply scaling to query and key layers.
        masked_softmax_fusion: Whether to use a kernel that fuses attention softmax with its mask.
        fp16_lm_cross_entropy: Whether to move the cross entropy unreduced loss calculation for lm head to fp16.
        share_embeddings_and_output_weights: Whether to share embeddings and output weights.
        enable_autocast: Whether to enable autocast for mixed precision.
        biobert_spec_option: BiobertSpecOption for the model.
        position_embedding_type: Type of position embedding used in the model.
        seq_length: Length of the input sequence.
        make_vocab_size_divisible_by: Make the vocabulary size divisible by this value.
        token_dropout: Whether to apply token dropout.
        use_attention_mask: Whether to use attention mask.
        use_esm_attention: Whether to use ESM attention.
        attention_softmax_in_fp32: Whether to use fp32 for attention softmax.
        optimizer_fn: Optional optimizer function for the model.
        parallel_output: Whether to use parallel output.
        rotary_base: Base value for rotary positional encoding.
        rotary_percent: Percentage of rotary positional encoding.
        seq_len_interpolation_factor: Interpolation factor for sequence length.
        get_attention_mask_from_fusion: Whether to get attention mask from fusion.
        nemo1_ckpt_path: Path to NEMO1 checkpoint.
        return_only_hidden_states: Whether to return only hidden states.
        loss_reduction_class: Loss reduction class for the model. Default to BERTMLMLossWithReduction.
    """

    # When overriding fields in a dataclass _always_ declare types: https://github.com/python/cpython/issues/123269
    model_cls: Type[AMPLIFYModelT] = AMPLIFYModel
    seq_length: int = 512
    num_layers: int = 24  # 32 for 350M, 24 for 120M
    hidden_size: int = 640  # 960 for 350M, 640 for 120M
    num_attention_heads: int = 10 # 15 for 350M, 10 for 120M
    ffn_hidden_size: int = 2560  # Transformer FFN hidden size. Usually 4 * hidden_size.
    hidden_dropout: float = 0  # AMPLIFY removes dropout from hidden layers and attention
    attention_dropout: float = 0.0  # AMPLIFY does not use attention dropout
    apply_residual_connection_post_layernorm: bool = False  # TODO: farhadr False is new default, True was BERT pub.
    layernorm_epsilon: float = 1.0e-5
    init_method_std: float = 0.02

    # embedding
    token_dropout: bool = True
    use_attention_mask: bool = True

    # core attention
    use_esm_attention: bool = False  # Skip ESM2 custom attention for TE acceleration. Still passes golden value test.
    attention_softmax_in_fp32: bool = False
    normalize_attention_scores: bool = False

    # From megatron.core.models.gpt.bert_model.GPTModel
    fp16_lm_cross_entropy: bool = False  # Move the cross entropy unreduced loss calculation for lm head to fp16
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 1
    position_embedding_type: PositionEmbeddingKinds = "rope"
    rotary_base: int = 10000
    rotary_percent: float = 1.
    
    #AMPLIFY specific configuration
    add_bias_linear: bool = False # AMPLIFY does not use bias in linear layers
    bias_swiglu_fusion: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    apply_rope_fusion: bool = True
    gated_linear_unit: bool = True
    activation_func: str = silu 
    # normalization: str = "RMSNorm"    # AMPLIFY uses RMSNorm instead of LayerNorm
    layernorm_zero_centered_gamma: bool = False # Zero centered gamma not supported for RMSNorm
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.amplify_bert_layer_with_transformer_engine_spec

    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False

    optimizer_fn: Optional[Callable[[MegatronBioBertModel], Optimizer]] = None
    # TODO (@skothenhill,@georgea) update to use the nemo2 checkpoint mixins
    #  support HF (requires weight interleaving on qkv layer) and nemo1 checkpoints ideally.
    nemo1_ckpt_path: str | None = None
    # The following checkpoint path is for nemo2 checkpoints. Config parameters not present in
    #  self.override_parent_fields will be loaded from the checkpoint and override those values here.
    initial_ckpt_path: str | None = None
    # TODO (@jstjohn) come up with a cleaner way in the biobert module to return user requested
    #  things as part of the workflow for inference and fine-tuning.
    return_embeddings: bool = False
    include_embeddings: bool = False
    skip_logits: bool = False
    return_only_hidden_states: bool = False  # return logits

    def __post_init__(self):
        """Check compatibility between biobert_spec_option and apply_query_key_layer_scaling post initialization."""
        super().__post_init__()
        if self.biobert_spec_option == BiobertSpecOption.amplify_bert_layer_with_transformer_engine_spec:
            self.apply_query_key_layer_scaling = False
            self.core_attention_override = ESM2TEDotProductAttention #TODO: ynashed: verify if this is needed
            if self.gated_linear_unit:
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
                # avoid RuntimeError due to misaligned operand
                multiple_of = 8
                self.ffn_hidden_size = int(2 * self.ffn_hidden_size / 3)
                self.ffn_hidden_size = multiple_of * ((self.ffn_hidden_size + multiple_of - 1) // multiple_of)
        else:
            raise ValueError(f"Unknown biobert_spec_option: {self.biobert_spec_option}")
