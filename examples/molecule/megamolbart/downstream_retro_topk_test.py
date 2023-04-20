# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import List, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import yaml
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from torch.utils.data import DataLoader

from bionemo.data.dataset_builder_utils import build_typed_dataset
from bionemo.data.molecule.augment import MoleculeInputTargetEnumeration
from bionemo.model.molecule.megamolbart import MegaMolBARTRetroModel
from bionemo.model.molecule.megamolbart.megamolbart_model_base import MegaMolBARTModelBase
from bionemo.model.utils import initialize_distributed_parallel_state, initialize_model_parallel

try:
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

from rdkit import Chem
from tqdm import tqdm

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def calculate_beam_metrics(sampled_smiles: List[List[str]], target_smiles, topk_beams: List[int] = [1]):
    """Helper method of run_topk_test. It calculates metrics of the topK test aas well as validity
    metrics of predicted samples given input and target SMILES.

    The sampled SMILES are assumed to be ranked in decreasing order according to some relevance score measure
    ie the confidence/probability of a sample

    TopK test measures the proportion of inputs for which their ground truth was within topK samples
    Args:
        sampled_smiles (List[List[str]]): list with the sampled SMILES per target being stored as list of str
        target_smiles (List[str]): canonicalized SMILES of ground truth
        topk_beams (List[int]): upper limits of topK test brackets

    Returns:
        dict, List[str], List[str], List[int]: dictionaries of topK test-related metrics and lists with SMILES samples
    """
    sampled_mols = [[Chem.MolFromSmiles(smi) for smi in sampled_smiles_i] for sampled_smiles_i in sampled_smiles]

    invalid = [[mol is None for mol in sampled_mols_i] for sampled_mols_i in sampled_mols]

    canonical_smiles = [["Unknown" if mol is None else Chem.MolToSmiles(mol, canonical=True) for mol in sampled_mols_i]
                        for sampled_mols_i in sampled_mols]
    idx_correct_smiles = [[idx for idx, smi in enumerate(canonical_smiles_i) if target_smiles[idx_i] == smi]
                          for idx_i, canonical_smiles_i in enumerate(canonical_smiles)]

    invalid_topk, topk = {}, {}
    min_idx_correct_smiles = [min(idx_correct_smiles_i) if len(idx_correct_smiles_i) else False
                              for idx_correct_smiles_i in idx_correct_smiles]
    for k in topk_beams:
        # calculates if the target smiles is within k sampled molecules
        topk[f'top_{k}'] = [min_idx < k if min_idx is not False else False for min_idx in min_idx_correct_smiles]
        # calculates number of invalid molecules sampled within first k
        invalid_topk[f'invalid_{k}'] = [sum(invalid_i[:k]) / k for invalid_i in invalid]

    metrics = {}
    for metric_name, metric_value in topk.items():
        metrics[metric_name] = sum(metric_value)
    for metric_name, metric_value in invalid_topk.items():
        metrics[metric_name] = sum(metric_value)

    return metrics, (sampled_smiles, canonical_smiles, idx_correct_smiles)


def run_topk_test(model: MegaMolBARTModelBase, dataloader: DataLoader, sampling_method: str, topk_beams: List[int],
                  limit_batches: Optional[int] = None, output_predictions: bool = False, sampling_kwargs: dict = {}) \
        -> Tuple[Dict[str, float], Dict[str, Dict[str, Union[str, List[str], List[int]]]]]:
    """
    Runs topK test given a sample (model) and calculates metrics per data iterated by dataloader.
    TopK test measures the proportion of inputs for which their ground truth target was within top K best samples

    The sampling method should output samples per input that are ranked in decreasing order according
    to some relevance score measure ie the confidence/probability of a prediction

    Args:
        model (MegaMolBARTModelBase): inference model that implements "sample_molecules" method
        dataloader (DataLoader): dataloader that iterates inputs and target SMILES
        sampling_method (str): sampling method specification
        topk_beams (List[int]): upper limits of topK test brackets
        limit_batches (Optional[int]): limit dataloader to a number of batches
        output_predictions (bool): should the predicted samples be returned?
        sampling_kwargs (dict): sampling-specific parameters passed to "sample" method.

        Please refer to the method get_sampling_token_fn in NeMo to see applicable sampling methods and
        corresponding sampling kwargs.
    Returns:
        dict, dict: dicts with topK test-related metrics and sampled SMILES
    """

    metrics = {}
    n_samples = 0
    preds_all = {}
    if limit_batches is not None:
        n_batches = limit_batches
    else:
        n_batches = len(dataloader)
    for i, batch in enumerate(tqdm(dataloader, desc="TopK test batch", total=n_batches)):
        tokens_enc, _, _, _, enc_mask, _ = model.process_global_batch(batch)
        target_smiles = batch['target_smiles']
        sampled_smiles = model.sample_molecules(tokens_enc, enc_mask, sampling_method=sampling_method,
                                                sampling_kwargs=sampling_kwargs)
        metrics_batch, (sampled_smiles, canonical_smiles, idx_correct_smiles) = calculate_beam_metrics(sampled_smiles,
                                                                                                       target_smiles,
                                                                                                       topk_beams)
        if output_predictions:
            for n, (product_ids, target, preds, preds_can, idx_correct) in enumerate(
                    zip(tokens_enc.tolist(), target_smiles, sampled_smiles, canonical_smiles, idx_correct_smiles)):
                product = model.tokenizer.ids_to_tokens(product_ids)
                product_smi = ''.join([s for s in product if s != model.tokenizer.pad_token])
                preds_all.update({n_samples: {'product_batch': ''.join(product), 'product_smi': product_smi,
                                              'target': target, 'predictions': preds,
                                              'predictions_canonized': preds_can, 'idx_correct': idx_correct
                                              }})
                n_samples += 1
        else:
            n_samples += len(target_smiles)
        if i == 0:
            for metric_name, metric_value in metrics_batch.items():
                metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in metrics_batch.items():
                metrics[metric_name].append(metric_value)
        if i == limit_batches:
            break

    for metric_name, metric_value in metrics.items():
        metrics[metric_name] = sum(metrics[metric_name]) / n_samples

    return metrics, preds_all


@timing
@hydra_runner(config_path="conf", config_name="downstream_retro_topk_test")
def main(cfg) -> None:
    """
    Main method that setups required components of the topK test (model, dataloader and its configuration,
    runs it and saves results
    """
    pl.seed_everything(cfg.model.data.seed)

    sampling_method = cfg.inference.get('sampling_method', None)
    sampling_kwargs = cfg.inference.get('sampling_kwargs', None)
    topk_beams = cfg.inference.get('topk_beams', [1])
    limit_batches = cfg.inference.get('limit_batches', None)
    beam_size = sampling_kwargs.beam_size
    augment = cfg.inference.get('augment', False)
    output_predictions = cfg.inference.get('output_predictions', False)
    batch_size = cfg.model.data.batch_size

    trainer = Trainer(**cfg.trainer)

    restore_path = cfg.model.downstream_task.get('restore_from_path', None)
    assert restore_path is not None, f'To run this test either a .nemo or .ckpt of the model should be provided'
    if restore_path.endswith('.nemo'):
        logging.info(f'Resuming training from nemo file: {restore_path}')
        model = MegaMolBARTRetroModel.restore_from(restore_path, trainer=trainer,
                                                   save_restore_connector=NLPSaveRestoreConnector())
        initialize_model_parallel(model)

    else:
        logging.info(f'Resuming training from checkpoint: {restore_path}')
        checkpoint_path = inject_model_parallel_rank(restore_path)
        trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=checkpoint_path)
        model = MegaMolBARTRetroModel.load_from_checkpoint(checkpoint_path, trainer=trainer)
    initialize_distributed_parallel_state(local_rank=0, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                          pipeline_model_parallel_split_rank=0)

    dataset = build_typed_dataset(dataset_paths=cfg.model.data.dataset_path,
                                  data_impl=cfg.model.data.data_impl, use_upsampling=cfg.model.data.use_upsampling,
                                  cfg=cfg.model.data, num_samples=None)

    collate_fn = MoleculeInputTargetEnumeration(tokenizer=model.tokenizer, seq_length=model._cfg.seq_length,
                                                pad_size_divisible_by_8=True,
                                                encoder_augment=augment, encoder_mask=False,
                                                decoder_augment=False, decoder_mask=False,
                                                canonicalize_input=False,
                                                input_name=cfg.model.data.input_name,
                                                target_name=cfg.model.data.target_name,
                                                ).collate_fn

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=cfg.model.data.num_workers, drop_last=False,
                            collate_fn=collate_fn)

    _reconfigure_microbatch_calculator(
        rank=0,  # This doesn't matter since it is only used for logging
        rampup_batch_size=None,
        global_batch_size=batch_size,
        micro_batch_size=batch_size,  # Make sure that there is no "grad acc" while decoding.
        data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
    )

    if limit_batches:
        n_samples = limit_batches * batch_size
    else:
        n_samples = len(dataset)

    results_path = os.path.join(cfg.model.data.output_fname)
    if ~os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    logging.info(f'Running topK for beam_size {beam_size} on {n_samples} samples from the MegaMolBARTRetro testset. \n'
                 f'The results are saved in {results_path}')

    topk_metrices, preds = run_topk_test(model=model, dataloader=dataloader,
                                         sampling_method=sampling_method,
                                         topk_beams=topk_beams,
                                         limit_batches=limit_batches,
                                         output_predictions=output_predictions,
                                         sampling_kwargs=sampling_kwargs)

    print("***************************")
    print(topk_metrices)
    print("***************************")

    filename_kwargs = '__'.join([k + str(v) for k, v in sampling_kwargs.items()])
    filename = f'results__{sampling_method}__{filename_kwargs}__batchsize{batch_size}__nsamples{n_samples}_aug{augment}'

    filename_metrices = filename + '.yaml'
    with open(os.path.join(results_path, filename_metrices), 'w') as file:
        yaml.safe_dump(topk_metrices, file)

    if output_predictions:
        filename_preds = filename + '__preds.yaml'
        with open(os.path.join(results_path, filename_preds), 'w') as file:
            yaml.safe_dump(preds, file)


if __name__ == '__main__':
    main()
