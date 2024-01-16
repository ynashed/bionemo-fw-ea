# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

import torch
from omegaconf import ListConfig
from pandas import DataFrame, Series
from pytorch_lightning.core import LightningModule

from bionemo.data.utils import pad_token_ids
from bionemo.model.utils import _reconfigure_inference_batch, initialize_model_parallel, restore_model


try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
    )

    HAVE_APEX: bool = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__: Sequence[str] = (
    "BaseEncoderDecoderInference",
    "M",
    "SeqsOrBatch",
    "Forward",
    "Inference",
    "to_sequences_ids",
    "HAVE_APEX",
)

SeqsOrBatch = Union[List[str], List[List[str]]]


class IdedSeqs(TypedDict):
    id: List[int]
    sequences: List[str]


class SI(TypedDict):
    sequence_ids: torch.Tensor


class Hidden(SI):
    hiddens: torch.Tensor
    mask: torch.Tensor


class Embedding(SI):
    embeddings: torch.Tensor


class Forward(Embedding, Hidden, total=False):
    """Output of BaseEncoderDecoderInference's forward()"""


class InferenceBase(TypedDict):
    hiddens: torch.Tensor
    mask: torch.Tensor
    embeddings: torch.Tensor
    sequence: List[str]


class Inference(InferenceBase, total=False):
    id: List[int]


# FIXME: add mask for all non-special tokens (add hiddens_tokens_only)
# TODO: add model-specific prepare_for_inference and release_from_inference methods
class BaseEncoderDecoderInference(LightningModule):
    '''
    Base class for inference.
    '''

    def __init__(
        self,
        cfg,
        model: Optional[Any] = None,
        freeze: bool = True,
        restore_path: Optional[str] = None,
        training: bool = False,
        adjust_config: bool = True,
        interactive: bool = False,
    ):
        super().__init__()

        self.cfg = cfg
        self._freeze_model = freeze
        self.adjust_config = adjust_config
        self.training = training
        self.interactive = interactive
        self.model = self.load_model(cfg, model=model, restore_path=restore_path)
        self._trainer = self.model.trainer
        self.tokenizer = self.model.tokenizer

        try:
            self.k_sequence: Optional[str] = cfg.model.data.data_fields_map.sequence
        except AttributeError:
            print(
                "WARNING: Missing key for extracting the sequence in batches! The forward method call will fail! "
                "Provide a valid configuration that has this value under model.data.data_fields_map.sequence."
            )
            self.k_sequence = None

        try:
            self.k_id: Optional[str] = cfg.model.data.data_fields_map.id
        except AttributeError:
            print(
                "WARNING: Missing key for extracting the sequence IDs in batches! The forward method call will fail! "
                "Provide a valid configuration that has this value under model.data.data_fields_map.id."
            )
            self.k_id = None

    def load_model(self, cfg, model: Optional[Any] = None, restore_path: Optional[str] = None) -> Any:
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            Loaded model
        """

        # load model class from config which is required to load the .nemo file

        if model is None:
            if restore_path is None:
                restore_path = cfg.model.downstream_task.restore_from_path
            model = restore_model(
                restore_path=restore_path, cfg=cfg, adjust_config=self.adjust_config, interactive=self.interactive
            )
        # move self to same device as loaded model
        self.to(model.device)

        # check whether the DDP is initialized
        initialize_model_parallel(model, interactive=self.interactive)

        if not self.interactive:
            # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
            _reconfigure_microbatch_calculator(
                rank=0,  # This doesn't matter since it is only used for logging
                rampup_batch_size=None,
                global_batch_size=1,
                micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
                data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
            )

        # Check for PEFT flag before calling `setup_optimizer_param_groups`
        if cfg.get('use_peft', False):  # skipped if use_peft is false or not present in config
            model.setup_optimizer_param_groups()
        elif self._freeze_model:  # only use encoder_frozen flag if not doing peft
            model.freeze()

        self.model = model

        return model

    def forward(self, batch: Dict[str, torch.Tensor]) -> Forward:
        """Forward pass of the model. Can return embeddings or hiddens, as required"""
        if self.k_sequence is None or self.k_id is None:
            raise ValueError(
                "Configuration used during initialization was invalid: it needs 2 keys. "
                "(1) It needs a key to identify the sequence in each batch (model.data.data_fields_map.sequence). "
                "(2) It needs a key to identify the sequence IDs in each batch (model.data.data_fields_map.id)."
            )
        sequences = batch[self.k_sequence]
        sequence_ids = batch[self.k_id]
        prediction_data = {"sequence_ids": sequence_ids}
        outputs = self.cfg.model.downstream_task.outputs
        # make sure we have a list
        if not isinstance(outputs, ListConfig):
            outputs = [outputs]

        # adjust microbatch size
        if not self.interactive:
            _reconfigure_inference_batch(global_batch_per_gpu=len(sequences))

        with torch.set_grad_enabled(self._freeze_model):
            for output_type in outputs:
                if output_type == 'hiddens':
                    hiddens, mask = self.seq_to_hiddens(sequences)
                    prediction_data["hiddens"] = hiddens
                    prediction_data["mask"] = mask
                elif output_type == 'embeddings':
                    prediction_data["embeddings"] = self.seq_to_embeddings(sequences)
                else:
                    raise ValueError(
                        f"Invalid prediction type: {self.cfg.model.downstream_task.prediction} "
                        f"For output type: {output_type}"
                    )

        return prediction_data

    def _tokenize(self, sequences: List[str]) -> torch.Tensor:
        """
        Model specific tokenization.
        Here <BOS> and <EOS> tokens are added for instance.

        Returns:
            token_ids (torch.Tensor, long): token ids
        """
        raise NotImplementedError("Please implement in child class")

    def tokenize(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize sequences.
        Returns:
            token_ids (torch.Tensor, long): token ids
            mask (torch.Tensor, long, float): boolean mask for padded sections
        """
        token_ids = self._tokenize(sequences=sequences)

        # Validate input sequences length
        if any(len(t) > self.model.cfg.seq_length for t in token_ids):
            raise ValueError(f'One or more sequence exceeds max length({self.model.cfg.seq_length}).')

        # Pad token ids (1/True = Active, 0/False = Inactive)
        token_ids, mask = pad_token_ids(
            token_ids,
            padding_value=self.tokenizer.pad_id,
            device=self.device,
        )

        return token_ids, mask

    def _detokenize(self, tokens_ids: List[List[str]]) -> List[str]:
        """
        Helper method for detokenize method
        Args:
            tokens_ids (list[list[str]]): a list with sequence of str stored as list

        Returns:
            sequences (list[str]): list of str corresponding to sequences
        """
        for i, cur_tokens_id in enumerate(tokens_ids):
            if self.tokenizer.eos_id in cur_tokens_id:
                idx = cur_tokens_id.index(self.tokenizer.eos_id)
                tokens_ids[i] = cur_tokens_id[:idx]
            else:
                tokens_ids[i] = [id for id in cur_tokens_id if id != self.tokenizer.pad_id]

        sequences = self.tokenizer.ids_to_text(tokens_ids)
        return sequences

    def detokenize(self, tokens_ids: torch.Tensor) -> SeqsOrBatch:
        """
        Detokenize a matrix of tokens into a list or nested list of sequences (i.e., strings).

        Args:
            tokens_ids (torch.Tensor, long): a matrix of token ids

        Returns:
            sequences (list[str] or list[list[str]]): list of sequences
        """
        tensor_dim = len(tokens_ids.size())
        supported_dims = [2, 3]

        tokens_ids = tokens_ids.cpu().detach().numpy().tolist()
        if tensor_dim == 2:
            # For instance, can correspond to the greedy search or topkp sampling where tensors with
            # predicted tokens ids have shape [batch_size, num_tokens_to_generate]
            return self._detokenize(tokens_ids=tokens_ids)
        elif tensor_dim == 3:
            # For instance, can correspond to the beam search with beam_size >1 where tensors with predicted tokens ids
            # have shape [batch_size, beam_size, num_tokens_to_generate]
            sequences = []
            for tokens_ids_i in tokens_ids:
                sequences.append(self._detokenize(tokens_ids=tokens_ids_i))
            return sequences
        else:
            raise ValueError(
                f'The shape of the tensor with token_ids is not supported. '
                f'Supported numbers of dims: {supported_dims}'
            )

    def seq_to_hiddens(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Transforms Sequences into hidden state.
        This class should be implemented in a child class, since it is model specific.
        This class should return only the hidden states, without the special tokens such as
         <BOS> and <EOS> tokens, for example.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        raise NotImplementedError("Please implement in child class")

    def hiddens_to_embedding(self, hidden_states: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        '''
        Transforms hidden_states into embedding.

        Args:
            hidden_states (torch.Tensor, float): hidden states
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            embeddings (torch.Tensor, float):
        '''
        # compute average on active hiddens
        lengths = enc_mask.sum(dim=1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError("Empty input is not supported (no token was proveded in one or more of the inputs)")

        embeddings = torch.sum(hidden_states * enc_mask.unsqueeze(-1), dim=1) / lengths

        return embeddings

    def seq_to_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """Compute hidden-state and padding mask for sequences.

        Params
            sequences: strings, input sequences

        Returns
            embedding array
        """
        # get hiddens and mask
        hiddens, enc_mask = self.seq_to_hiddens(sequences)
        # compute embeddings from hiddens
        embeddings = self.hiddens_to_embedding(hiddens, enc_mask)

        return embeddings

    def hiddens_to_seq(self, hidden_states: torch.Tensor, enc_mask: torch.Tensor, **kwargs) -> SeqsOrBatch:
        """
        Transforms hidden state into sequences (i.e., sampling in most cases).
        This class should be implemented in a child class, since it is model specific.
        This class should return the sequence with special tokens such as
         <BOS> and <EOS> tokens, if used.

        Args:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections

        Returns:
            sequences (list[str] or list[list[str]]): list of sequences
        """
        raise NotImplementedError("Please implement in child class")

    @property
    def supported_sampling_methods(self) -> List[str]:
        """
        Returns a list of supported sampling methods.
        Example:
            ["greedy-perturbate", "beam-search"]
        """
        return list(self.default_sampling_kwargs.keys())

    @property
    def default_sampling_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dict of default sampling kwargs per sampling method.
        Example:
            {
                "greedy-perturbate": {"scaled_radius": 1, "smis": []},
                "beam-search": {"beam_size": 5, "beam_alpha": 0.6, "smis": []},
            }

        Should be overridden in child class if sampling is supported.
        """
        return {}

    def sample(
        self,
        num_samples: int = 1,
        return_embedding: bool = False,
        sampling_method: Optional[str] = None,
        **sampling_kwarg,
    ) -> Union[SeqsOrBatch, Tuple[SeqsOrBatch, torch.Tensor]]:
        """
        Sample from the model given sampling_method.

        Args:
            num_samples (int): number of samples to generate (depends on sampling method)
            return_embedding (bool): return embeddings corresponding to each of the samples in addition to the samples
            sampling_method (str): sampling method to use. Should be replaced with default sampling method in child class
            sampling_kwarg (dict): kwargs for sampling method. Depends on the sampling method.
        """
        raise NotImplementedError(f"Sampling is not supported in this class ({self.__class__.__name__})")

    def __call__(self, sequences: Union[Series, IdedSeqs, List[str]]) -> Inference:
        """
        Computes embeddings for a list of sequences.
        Embeddings are detached from model.

        Params
            sequences: Pandas Series containing a list of strings or or a list of strings (e.g., SMILES)

        Returns
            embeddings
        """
        seqs, ids = to_sequences_ids(sequences)

        hiddens, enc_mask = self.seq_to_hiddens(seqs)
        embeddings = self.hiddens_to_embedding(hiddens, enc_mask)

        result_dict: Inference = {
            "embeddings": embeddings.float().detach().clone(),
            "hiddens": hiddens.float().detach().clone(),
            "mask": enc_mask.detach().clone(),
            "sequence": seqs,
        }
        if ids is not None:
            result_dict["id"] = ids

        return result_dict


M = TypeVar('M', bound=BaseEncoderDecoderInference)
"""Generic type for any model that implements :class:`BaseEncoderDecoderInference`.
"""


def to_sequences_ids(
    sequences: Union[DataFrame, Series, IdedSeqs, List[str]]
) -> Tuple[List[str], Optional[List[int]]]:
    """Converts supported sequence data formats into a simple list of sequences and their ids.

    The input must be one of:
        - Pandas Series containing a list of strings
        - A dictionary with `id` and `sequence`, int IDs along with each string sequence, respectively
        - A DataFrame with columns `id` and `sequence`
        - a list of strings (e.g., SMILES)

    The returned output will be a copy of the sequences and their accompanying IDs. If there are no IDs,
    then the returned IDs value is `None`.
    """
    ids: Optional[List[int]] = None
    if isinstance(sequences, Series):
        seqs: List[str] = sequences.tolist()
    elif isinstance(sequences, Dict):
        ids = sequences["id"]
        seqs = sequences["sequence"]
    elif isinstance(sequences, DataFrame):
        ids = sequences.loc["id"].values.tolist()
        seqs = sequences.loc["sequence"].values.tolist()
    else:
        seqs = list(sequences)
    return seqs, ids
