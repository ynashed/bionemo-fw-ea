# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

import logging
import torch
from torch.nn.functional import pad
from pandas import Series
from typing import List, Union
from omegaconf import OmegaConf

from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import (NLPDDPStrategy,
                                                      NLPSaveRestoreConnector)
from nemo.utils.app_state import AppState

from bionemo.data.utils import pad_token_ids
from bionemo.data import MoleculeEnumeration
from bionemo.model.molecule.megamolbart import MegaMolBARTModel


log = logging.getLogger(__name__)
__all__ = ["NeMoMegaMolBARTWrapper", "MegaMolBARTValidationInferenceWrapper"]


# FIXME: switch to be based on BaseEncoderDecoderInference

class NeMoMegaMolBARTWrapper():
    '''
    Implements functions to infer using MegaMolBART model
    '''

    def __init__(self,
                 model_cfg=None,
                 random_weights=False) -> None:
        super().__init__()

        if model_cfg is None:
            # TODO: Create a default global variable for this
            log.info('Loading default configuration...')
            model_cfg = OmegaConf.load(
                '/workspace/bionemo/examples/bionemo/conf/infer.yaml')
        if random_weights:
            model_cfg['model']['model_path'] = None

        self.model = self.load_model(model_cfg)
        self.cfg = self.model._cfg
        self.max_seq_len = self.cfg.max_position_embeddings
        self.tokenizer = self.model.tokenizer

        pad_size_divisible_by_8 = True if self.cfg.masked_softmax_fusion else False
        self.mol_enum = MoleculeEnumeration(tokenizer=self.tokenizer,
                                            seq_length=self.cfg.seq_length,
                                            pad_size_divisible_by_8=pad_size_divisible_by_8,
                                            **self.cfg.data)
        self.mol_enum.encoder_mask = False

        self.mol_enum.encoder_augment = False
        self.mol_enum.encoder_mask = False
        self.mol_enum.canonicalize_input = False
        self.mol_enum.decoder_augment = False
        self.mol_enum.decoder_mask = False
        self.mol_enum.mask_prob = 0

    def _tokenize(self, smis: List[str]):
        """
        MegaMolBART expects input/output format:
        
        encoder input ids - [tokens] (without <BOS> and <EOS>)
        decoder input ids - <BOS> + [tokens]
        decoder output ids - [tokens] + <EOS>
        """
        tokens = [self.tokenizer.text_to_tokens(s) for s in smis]
        token_ids = [self.tokenizer.token_to_ids(t) for t in tokens]

        # Pad token ids (1/True = Active, 0/False = Inactive)
        token_ids, encoder_mask = pad_token_ids(
            token_ids, 
            padding_value=self.tokenizer.pad_id, 
            pad_size_divisible_by=8 if self.pad_size_divisible_by_8 else 1,  
            dtype=torch.int64,
            # FIXME: use model.device
            device="cuda",
            )

        return token_ids, encoder_mask

    def _transform(self, smis):
        '''
        Transforms SMILES into hidden state.

        Args:
            smis (list[str]): list of SMILES strings

        Returns:
            tokens_enc (torch.Tensor, long): token ID values for samples
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''

        tokens_enc, enc_mask = self._tokenize(smis)
        hidden_states = self.model.encode(tokens_enc, enc_mask)

        return hidden_states, enc_mask

    def load_model(self, model_cfg):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            MegaMolBART trained model
        # """
        torch.set_grad_enabled(False)

        # trainer required for restoring model parallel model
        trainer = Trainer(
            plugins=[],
            devices=1,
            accelerator='gpu',
            precision=32, #TODO: Run benchmark to verify this value has no or
            strategy=NLPDDPStrategy(),
            #                     minimum impact on KPIs.
        )

        app_state = AppState()
        if model_cfg.model.model_path is not None:
            model = MegaMolBARTModel.restore_from(
                restore_path=model_cfg.model.model_path,
                trainer=trainer,
                save_restore_connector=NLPSaveRestoreConnector(),
            )
        else:
            # Initialize with random weights
            cfg = OmegaConf.load(
                '/workspace/bionemo/examples/bionemo/conf/pretrain_base.yaml')
            cfg.model.num_layers=6
            cfg.model.hidden_size=512
            cfg.model.num_attention_heads=8
            cfg.model.precision = cfg.trainer.precision

            model = MegaMolBARTModel(cfg.model, trainer)

        model.freeze()

        return model

    def smis_to_hidden(self, smis: List[str]):
        """Compute hidden-state and padding mask for smiles.

        Params
            smi: string, input SMILES molecule

        Returns
            hidden-state array and boolean mask
        """
        if isinstance(smis, str):
            smis = [smis]

        hidden_states, enc_masks = self._transform(smis)
        return hidden_states, enc_masks

    def smis_to_embedding(self, smis: List[str]):
        """Computes embedding and padding mask for smiles.

        Params
            smi: string, input SMILES molecule

        Returns
            hidden-state array and boolean mask
        """
        if isinstance(smis, str):
            smis = [smis]

        hiddens, enc_masks = self.smis_to_hidden(smis)
        # compute average on active hiddens
        lengths = enc_masks.sum(dim=1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError("Empty input is not supported (no token was proveded in one or more of the inputs)")

        emb = torch.sum(hiddens*enc_masks.unsqueeze(-1), dim=1) / lengths

        return emb

    def hidden_to_smis(self, hidden_states, enc_mask):

        predicted_tokens_ids, _ = self.model.decode(None,
                                                    enc_mask,
                                                    self.cfg.max_position_embeddings,
                                                    enc_output=hidden_states)

        predicted_tokens_ids = predicted_tokens_ids.cpu().detach().numpy().tolist()
        for i, predicted_token_id in enumerate(predicted_tokens_ids):
            if self.tokenizer.eos_id in predicted_token_id:
                idx = predicted_token_id.index(self.tokenizer.eos_id)
                predicted_tokens_ids[i] = predicted_token_id[:idx]
            else:
                predicted_tokens_ids[i] = [id for id in predicted_token_id if id != self.tokenizer.pad_id]

        smis = self.tokenizer.ids_to_text(predicted_tokens_ids)

        return smis

    def sample(self,
               smis,
               num_samples=10,
               return_embedding=False,
               sampling_method='greedy-perturbate',
               sampling_kwarg={'scaled_radius': 1, 'topk': 10}):
        """
        Sample from model given hidden states and mask
        """
        hidden_states, enc_masks = self.smis_to_hidden(smis)

        if sampling_method == 'greedy-perturbate':
            scaled_radius = sampling_kwarg['scaled_radius']
            sample_masks = enc_masks.repeat_interleave(num_samples, 0)
            perturbed_hiddens = hidden_states.repeat_interleave(num_samples, 0)
            perturbed_hiddens = perturbed_hiddens + (scaled_radius * torch.randn(perturbed_hiddens.shape).to(perturbed_hiddens.device))

            samples = self.hidden_to_smis(perturbed_hiddens, sample_masks)
            if return_embedding:
                embs = torch.mean(perturbed_hiddens, dim=1)
        else:
            raise ValueError(f'Invalid samping method {sampling_method}')

        if return_embedding:
            return samples, embs
        else:
            return samples

# TODO refactor this wrapper's functionality into NeMoMegaMolBARTWrapper
class MegaMolBARTValidationInferenceWrapper:
    """Inference wrapper for MegaMolBART for use in the training loop"""
    
    def __init__(self, megamolbart: MegaMolBARTModel, cfg) -> None:
        self.mmb = megamolbart
        self.cfg = cfg
        self.microbatch_size = self.cfg.model.micro_batch_size
        
    def _pad_tensor(self, t: torch.Tensor, n: int) -> torch.Tensor:
        """
        Pads first dimension of tensor to nearest multiple of n, rounding up. Zero-fills padded elements
        
        Params
            t: Input tensor
            n: t will be padded to nearest multiple of n, rounding up
        
        Returns
            Padded tensor or same tensor if t.shape[0] % n == 0
        """
        length = t.shape[0]
        if length % n != 0:
            m = (length // n) + 1
            new_length = m * n
            diff = new_length - length
            padder = [0 for i in range(t.ndim * 2)]
            # pad the first dimension only
            padder[-1] = diff
            z = pad(t, padder)
            return z
        else:
            return t
        
    def _tokenize(self, smis: List[str]):
        """
        Tokenizes a list of SMILES strings
        
        Params
            smis: List of strings
            
        Returns
            (token_ids, encoder_mask)
        """
        tokens = [self.mmb.tokenizer.text_to_tokens(s) for s in smis]
        token_ids = [self.mmb.tokenizer.token_to_ids(t) for t in tokens]

        pad_length = max([len(seq) for seq in token_ids])
        encoder_mask = [([1] * len(seq)) + ([0] * (pad_length - len(seq))) for seq in token_ids]
        token_ids = [seq + ([self.mmb.tokenizer.pad_id] * (pad_length - len(seq))) for seq in token_ids]

        token_ids = torch.tensor(token_ids, dtype=torch.int64)
        encoder_mask = torch.tensor(encoder_mask,
                                    dtype=torch.int64,
                                    device=token_ids.device)

        return token_ids, encoder_mask
    
    def _smis_to_embedding(self, smis: Union[List[str], str]) -> torch.Tensor:
        """
        Computes embedding for smiles.

        Params
            smi: String or list of Strings

        Returns
            embeddings
        """
        if type(smis) is str:
            smis = [smis]
        
        with torch.no_grad():
            # Calling model.encode() interferes with global microbatch settings and raises errors when reconfigure_microbatch=True.
            # Workaround: pad and batch smiles to match microbatch size before calling model.encode() with reconfigure_microbatch=False
            tokens_enc, enc_mask = self._tokenize(smis)
            tokens_enc_padded, enc_mask_padded = self._pad_tensor(tokens_enc, self.microbatch_size), self._pad_tensor(enc_mask, self.microbatch_size)
            tokens_enc_split, enc_mask_split = torch.split(tokens_enc_padded, self.microbatch_size, dim=0), torch.split(enc_mask_padded, self.microbatch_size, dim=0)
            
            hidden_states_chunked = [self.mmb.encode(tokens_chunk, mask_chunk, reconfigure_microbatch=False) for tokens_chunk, mask_chunk in zip(tokens_enc_split, enc_mask_split)]
            hidden_states = torch.cat(hidden_states_chunked, dim=0)
            hidden_states = hidden_states[:tokens_enc.shape[0], ...]
            
            enc_mask = enc_mask.to(hidden_states.device)
                        
            lengths = enc_mask.sum(dim=1, keepdim=True)
            if (lengths==0).any():
                raise ValueError("Empty input is not supported (no token was proveded in one or more of the inputs)")
            
            embeddings = torch.sum(hidden_states*enc_mask.unsqueeze(-1), dim=1) / lengths
        
        return embeddings
    
    def __call__(self, smiles_series: Series) -> torch.Tensor:
        """
        Computes embeddings for a list of SMILES
        
        Params
            smiles: Pandas Series containing a list of SMILES strings
            
        Returns
            embeddings
        """
        smiles = smiles_series.tolist()
        return self._smis_to_embedding(smiles).float().detach().clone()
