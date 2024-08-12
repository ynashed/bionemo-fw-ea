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
'''
How to adapt these tests:

1) Need to hook up our own pretraining workflow. In the code below, we do this via subproc and CLI. Is this still best practice?
    a) use the structure in sub-packages/bionemo-geneformer/tests/bionemo/geneformer/test_model.py:test_geneformer_nemo1_v_nemo2_inference_golden_values
    b) might need to look at utilities for setup/teardown to make sure the distributed stuff is handled correctly.
2) Need to inject the callbacks either via CLI or by manually inserting them here.
3) How do we want this to work for other modules? Lots of code could be duplicated here which makes it a challenge.
4) is this the right set of code to do this on?

'''
import math
import pytest
import pathlib
from typing import Literal, List, Tuple
from bionemo import geneformer
from nemo.collections import llm
# Do we want to re-export stuff in api?
from bionemo.geneformer.api import GeneformerConfig
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.core.utils.batching_utils import pad_token_ids
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.core.utils.dtypes import get_autocast_dtype
from nemo.lightning.pytorch import callbacks as nl_callbacks
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo import lightning as nl
from torch.nn import functional as F
import torch

from nemo.lightning import resume
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule

bionemo2_root: pathlib.Path = (
    # geneformer module's path is the most dependable --> don't expect this to change!
    pathlib.Path(geneformer.__file__)
    # This gets us from 'sub-packages/bionemo-geneformer/src/bionemo/geneformer/__init__.py' to 'sub-packages/bionemo-geneformer'
    .parent.parent.parent.parent
    # From here, we want to get to the root of the repository: _before_ sub-packages/
    .parent.parent
).absolute()

assert bionemo2_root != pathlib.Path("/")
data_path: pathlib.Path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/processed_data"

MODEL_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"
USE_TE: bool = False  # TODO use this for high level decisions around whether we're ready to switch to TE
CELLS_FOR_TEST: List[List[str]] = [
    [
        "ENSG00000288623",
        "ENSG00000288658",
        "ENSG00000288681",
        "ENSG00000288698",
        "ENSGR0000002586",
        "ENSGR0000124333",
        "ENSGR0000124334",
        "ENSGR0000167393",
        "ENSGR0000168939",
        "ENSGR0000169084",
    ],
    [
        "ENSG00000259900",
        "ENSG00000259916",
        "ENSG00000259956",
        "ENSG00000259958",
        "ENSG00000259991",
        "ENSG00000260001",
        "ENSG00000260007",
        "ENSG00000260027",
        "ENSG00000260040",
        "ENSG00000260045",
        "ENSG00000260092",
        "ENSG00000260099",
        "ENSG00000260119",
    ],
    [
        "ENSG00000269743",
        "ENSG00000269746",
        "ENSG00000269748",
        "ENSG00000269753",
        "ENSG00000269754",
        "ENSG00000269755",
        "ENSG00000269759",
        "ENSG00000269766",
        "ENSG00000269773",
        "ENSG00000269781",
        "ENSG00000269782",
        "ENSG00000269783",
        "ENSG00000269790",
        "ENSG00000269791",
        "ENSG00000269795",
    ],
]

def _apply_tokenizer(tokenizer, sequences: List[List[str]], device) -> List[torch.Tensor]:
    # parent pulls the tokenizer from the loaded model.
    try:
        token_ids = [
            torch.tensor(
                [tokenizer.class_id] + [tokenizer.token_to_id(gene_symbol) for gene_symbol in gene_symbols],
                device=device,
                dtype=torch.long,
            )
            for gene_symbols in sequences
        ]
    except TypeError as e:
        invalid_tokens = {gene_symbol for gene_symbols in sequences for gene_symbol in gene_symbols} - set(
            tokenizer.vocab.keys()
        )
        raise ValueError(
            f"Unknown token in gene symbols. Please filter genes for those present in self.tokenizer:\n{invalid_tokens}"
        ) from e
    return token_ids

def _batched_tokenizer(
    tokenizer, sequences: List[List[str]], device, seq_length: int = 2048, dynamic_padding: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize sequences.
    Returns:
        token_ids (torch.Tensor, long): token ids
        mask (torch.Tensor, long, float): boolean mask for padded sections
    """
    token_ids = _apply_tokenizer(tokenizer=tokenizer, sequences=sequences, device=device)

    # Validate input sequences length
    if any(len(t) > seq_length for t in token_ids):
        raise ValueError(f"One or more sequence exceeds max length({seq_length}).")

    # Set fixed padding when dynamic padding is disabled
    if not dynamic_padding:
        padding_length = seq_length
    else:
        padding_length = None
    # Pad token ids (1/True = Active, 0/False = Inactive)
    token_ids, mask = pad_token_ids(
        token_ids,
        padding_value=tokenizer.pad_id,
        padding_len=padding_length,
        device=device,
    )

    return token_ids, mask

class _DummyDataSet(torch.utils.data.Dataset):
    def __init__(self, cells: List[List[str]], tokenizer):
        input_ids, mask = _batched_tokenizer(tokenizer, cells, device=torch.device("cuda"))
        self.input_ids = input_ids
        self.mask = mask
        assert len(self.input_ids) == len(self.mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"text": self.input_ids[idx], "attention_mask": self.mask[idx]}

seq_length = 2048 # NOTE(@skothenhill) decrease this if there are memory issues in CI
@pytest.fixture
def geneformer_config():
    """ Setups the default geneformer config taken from pretrain.py. Update as needed. """
    autocast_dtype = get_autocast_dtype(MODEL_PRECISION)
    return GeneformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=seq_length,
        fp16=autocast_dtype == torch.float16,
        bf16=autocast_dtype == torch.bfloat16,
        fp32_residual_connection=False,
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=False,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,
        fp16_lm_cross_entropy=False,
        params_dtype=autocast_dtype,
        pipeline_dtype=autocast_dtype,
        autocast_dtype=autocast_dtype,
        gradient_accumulation_fusion=False,
        layernorm_zero_centered_gamma=False,
        layernorm_epsilon=1.0e-12,
        activation_func=F.gelu,
        qk_layernorm=False,
        apply_residual_connection_post_layernorm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,
        biobert_spec_option=BiobertSpecOption.bert_layer_with_transformer_engine_spec
        if USE_TE
        else BiobertSpecOption.bert_layer_local_spec,
        nemo1_ckpt_path=None,
        return_only_hidden_states=True,  # This is what we did in nemo1 for inference
    )

# TODO (skothenhill) which dataloader makes more sense for this test?
#   - dummy will have the loss descending faster, which is good. But we need it in a DataLoader, no bueno.
#   - real will be more representative of the actual use case, but harder to maintain the test.
def make_dummy_dataloader(tokenizer):
    """ Dummy dataloader for testing without requiring the full memmap. """
    cells = CELLS_FOR_TEST
    dataloader = torch.utils.data.DataLoader(_DummyDataSet(cells, tokenizer), batch_size=3, num_workers=0)
    return dataloader

def make_real_datamodule(tokenizer, seq_length, median_dict, devices, pipeline_model_parallel_size, data_path):
    from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=data_path / 'train',
        val_dataset_path=data_path / 'val',
        test_dataset_path=data_path / 'test',
        random_token_prob=0.1,  # this is the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=3,
        global_batch_size=3 * int(devices / pipeline_model_parallel_size),
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=True,
        pin_memory=False,
        num_workers=2,
    )
    return data

# TODO (@skothenhill) How can we adapt this into a test harness?
def test_geneformer_stop_and_go(
    geneformer_config: GeneformerConfig, seed: int = 42
):
    # Ensure the test data exists.
    data_error_str = "Please download test data with:\n`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    # Configuration stuff.
    data_dir = pathlib.Path(data_path)
    train_data_path = data_dir / "train"
    val_check_interval = 100
    lr=1e-4
    num_steps = 100000
    cosine_rampup_frac = 0.1
    cosine_hold_frac = 0.05
    root_dir = pathlib.Path("/workspace/bionemo2/results")

    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_error_str}")

    # Typical setup stuff. Do we need this with the test data? Find this out.
    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            pass
        case _:
            raise ValueError("Preprocessing must have failed.")

    
    # Taken from default argparse.
    module = BioBertLightningModule(config=geneformer_config, tokenizer=tokenizer,
            optimizer=MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=lr,
                optimizer="adam",
                use_distributed_optimizer=True,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=num_steps,
                min_lr=lr / 100,
                warmup_steps=int(math.ceil(num_steps * cosine_rampup_frac)),
                interval="step",
                monitor="val_loss",
                constant_steps=int(math.ceil(num_steps * cosine_hold_frac)),
            ),
        ),)
    data_module = make_real_datamodule(tokenizer, seq_length, median_dict, devices=1, pipeline_model_parallel_size=1, data_path=data_path)

    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=True,
        monitor="val_loss",
        save_top_k=1,
        every_n_train_steps=val_check_interval,
        enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
    )

    nemo_logger: NeMoLogger = NeMoLogger(
        dir=str(root_dir),
        name='geneformer-stopngo',
        tensorboard=None,
        wandb=None,
        ckpt=checkpoint_callback
    )

    # TODO (@skothenhill) Also impacts the DataModule? Should we be thinking about this?
    tp_size, pp_size, devices = 1, 1, 1
    micro_batch_size = 3
    seq_len = 16 # Full size is 2048
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
        val_check_interval=val_check_interval,
        data_sampler=nl.MegatronDataSampler(
            micro_batch_size=micro_batch_size,
            global_batch_size=micro_batch_size * int(devices / pp_size),
            seq_len=seq_len,
        ),
    )
    trainer = nl.Trainer(
        devices=tp_size * pp_size,
        accelerator="gpu",
        strategy=strategy,
        num_nodes=1,
        callbacks=None, # is this what I need for stop and go?
        plugins=nl.MegatronMixedPrecision(precision=MODEL_PRECISION, amp_O2=False),
    )
    llm.train(module, data_module, trainer, 
        log=nemo_logger,
        resume=resume.AutoResume(
            path=None,  # Overrides the path found by resume_if_exists when set.
            resume_if_exists=False,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        )
    )
    # TODO (@skothenhill) setup a callback that throws an exception or something similar after saving a checkpoint.
    # TODO (@skothenhill) setup the Trainer again, but this time use resume_if_exists=True
    # TODO (@skothenhill) Factor out the configuration code into a fixture, only keep strategy, trainer and llm.train here.
    # NOTE (@skothenhill) There are some variants of this we will want to test, such as: restoring from a direct path, using use_nemo_ckpt_io, changing parallelisms.
    # NOTE (@skothenhill) Ensure that teardown is handled correctly.


    llm.train(module, data_module, trainer, 
        log=nemo_logger,
        resume=resume.AutoResume(
            path=None,  # Overrides the path found by resume_if_exists when set.
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        )
    )






'''
## OLD CODE FROM BIONEMO 1 STARTS HERE
import os
import subprocess
from typing import List, Literal, Tuple, TypedDict

import pytest
import torch


BIONEMO_HOME = os.environ["BIONEMO_HOME"]
TEST_DATA_DIR = os.path.join(BIONEMO_HOME, "examples/tests/test_data")


class StopAndGoTestParams(TypedDict):
    script_path: str
    metadata_keys: List[str]


# possible values for metadata_list defined in getter_function_map in bionemo/callbacks/testing_callbacks.py
# new values can be defined there by adding new getter functions
TEST_PARAMS: List[StopAndGoTestParams] = [
    {"script_path": "examples/protein/openfold/train.py", "metadata_keys": ["learning_rate", "global_step"]},
    {"script_path": "examples/dna/dnabert/pretrain.py", "metadata_keys": ["learning_rate", "global_step"]},
    {"script_path": "examples/singlecell/geneformer/pretrain.py", "metadata_keys": ["learning_rate", "global_step"]},
    {"script_path": "examples/molecule/molmim/pretrain.py", "metadata_keys": ["learning_rate", "global_step"]},
    {"script_path": "examples/protein/esm2nv/pretrain.py", "metadata_keys": ["learning_rate", "global_step"]},
]

TRAINING_SCRIPTS_PATH = [params["script_path"] for params in TEST_PARAMS]
METADATA_LIST = [params["metadata_keys"] for params in TEST_PARAMS]


@pytest.fixture
def train_args():
    return {
        "trainer.devices": torch.cuda.device_count(),
        "trainer.num_nodes": 1,
        "trainer.max_steps": 8,
        "trainer.val_check_interval": 2,  # check validation set every 2 training batches
        "trainer.limit_val_batches": 1,  # run validation for 2 validation batches
        "model.data.val.use_upsampling": False,
        "trainer.limit_test_batches": 1,
        "model.data.test.use_upsampling": True,
        "exp_manager.create_wandb_logger": False,
        "exp_manager.create_tensorboard_logger": False,
        "model.micro_batch_size": 2,
    }


@pytest.fixture
def data_args():
    return {
        "model.data.dataset.train": "x000",
        "model.data.dataset.val": "x000",
        "model.data.dataset.test": "x000",
    }


def get_data_overrides(script_or_cfg_path: str) -> str:
    """Replace datasets with smaller samples included in the repo

    Based on the script/config file provided, checks what kind of task
    the script performs and selects compatible data sample from test data.
    Returns string that can be appended to the python command for launching the script
    """
    DATA = " ++model.data"
    MAIN = f"{DATA}.dataset_path={TEST_DATA_DIR}/%s"
    DOWNSTREAM = f" ++model.dwnstr_task_validation.dataset.dataset_path={TEST_DATA_DIR}/%s"

    root, domain, model, *conf, script = script_or_cfg_path.split("/")
    assert root == "examples" and model in (
        "megamolbart",
        "esm1nv",
        "esm2nv",
        "prott5nv",
        "downstream",
        "openfold",
        "dnabert",
        "diffdock",
        "molmim",
        "geneformer",
    ), "update this function, patterns might be wrong"

    task = {
        "molecule": "physchem/SAMPL",
        "protein": "downstream",
        "dna": "downstream",
        "singlecell": "downstream",
    }
    if conf == ["conf"]:
        if model in ("megamolbart", "openfold", "molmim"):
            return ""
        elif model == "geneformer":
            return MAIN % "singlecell"
        else:
            return MAIN % f"{domain}/{task[domain]}/test/x000"

    if "retro" in script:
        return MAIN % "reaction"
    elif model == "geneformer":
        return (
            # This is what we run inference on when running infer.py. This is not checked or used during pretraining.
            f" {DATA}.dataset_path={TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data/test"
            # The following three paths are used for pretrain.py, but also are required to support model loading currently when running inference.
            f" {DATA}.train_dataset_path={TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data/train"
            f" {DATA}.val_dataset_path={TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data/val"
            f" {DATA}.test_dataset_path={TEST_DATA_DIR}/cellxgene_2023-12-15_small/processed_data/test"
        )
    elif model == "openfold":
        return MAIN % "openfold_data/processed_sample"
    elif model == "diffdock":
        return (
            f" ++data.split_train={TEST_DATA_DIR}/molecule/diffdock/splits/split_train"
            f" ++data.split_val={TEST_DATA_DIR}/molecule/diffdock/splits/split_val"
            f" ++data.split_test={TEST_DATA_DIR}/molecule/diffdock/splits/split_test"
            f" ++data.cache_path={TEST_DATA_DIR}/molecule/diffdock/data_cache"
        )
    elif "downstream" in script:
        if model == "dnabert":
            fasta_directory = os.path.join(TEST_DATA_DIR, "dna/downstream")
            fasta_pattern = fasta_directory + "/test-chr1.fa"
            splicesite_overrides = (
                f"++model.data.fasta_directory={fasta_directory} "
                "++model.data.fasta_pattern=" + fasta_pattern + " "
                f"++model.data.train_file={fasta_directory}/train.csv "
                f"++model.data.val_file={fasta_directory}/val.csv "
                f"++model.data.predict_file={fasta_directory}/test.csv "
            )
            return splicesite_overrides
        else:
            return MAIN % f"{domain}/{task[domain]}"
    elif model == "dnabert":
        DNABERT_TEST_DATA_DIR = os.path.join(BIONEMO_HOME, "examples/dna/dnabert/data/small-example")
        dnabert_overrides = (
            f"++model.data.dataset_path={DNABERT_TEST_DATA_DIR} "
            "++model.data.dataset.train=chr1-trim-train.fna "
            "++model.data.dataset.val=chr1-trim-val.fna "
            "++model.data.dataset.test=chr1-trim-test.fna "
        )
        return dnabert_overrides
    elif model == "esm2nv" and "infer" not in script:
        UNIREF_FOLDER = "uniref202104_esm2_qc_test200_val200"
        esm2_overwrites = MAIN % UNIREF_FOLDER + DOWNSTREAM % f"{domain}/{task[domain]}"
        return esm2_overwrites

    else:
        return (MAIN + DOWNSTREAM) % (domain, f"{domain}/{task[domain]}")


def get_train_args_overrides(script_or_cfg_path, train_args):
    root, domain, model, *conf, script = script_or_cfg_path.split("/")
    if model == "openfold":
        train_args["model.micro_batch_size"] = 1
        train_args["model.train_ds.num_workers"] = 1
        train_args["model.train_sequence_crop_size"] = 16
        # do not use kalign as it requires third-party-download and it not essential for testing
        train_args["model.data.realign_when_required"] = False
    elif model == "diffdock":
        # Use size aware batch sampler, and set the size control to default
        train_args["model.micro_batch_size"] = 2
        train_args["model.estimate_memory_usage.maximal"] = "null"
        train_args["model.max_total_size"] = "null"
        train_args["model.tensor_product.type"] = "fast_tp"

    return train_args


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("script_path, metadata_keys", list(zip(TRAINING_SCRIPTS_PATH, METADATA_LIST)))
def test_stop_and_go(script_path: str, metadata_keys: List[str], train_args, data_args, tmp_path):
    data_str = get_data_overrides(script_path)
    train_args = get_train_args_overrides(script_path, train_args)
    # add kill-after-checkpoint and metadata-save callbacks for first run
    train_args["create_kill_after_signal_callback"] = True
    train_args["kill_after_signal_callback_kwargs.metadata_path"] = tmp_path
    train_args["create_metadata_save_callback"] = True
    train_args["metadata_save_callback_kwargs.metadata_path"] = tmp_path

    cmd = f"python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} " + " ".join(
        f"++{k}={v}" for k, v in train_args.items()
    )
    cmd = cmd + f' "++metadata_save_callback_kwargs.metadata_keys={metadata_keys}"'
    # TODO(dorotat) Trye to simplify when data-related utils for ESM2 are refactored
    if "esm2" not in script_path and "dnabert" not in script_path:
        cmd += " " + " ".join(f"++{k}={v}" for k, v in data_args.items())
    print(cmd)
    # run initial training run to save a checkpoint and some metadata and kill the job afterwards
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    error_out = process_handle.stderr.decode("utf-8")
    assert process_handle.returncode == 0, f"Initial training command failed:\n{cmd}\n Error log:\n{error_out}"

    # assert that metadata was saved correctly
    assert os.path.isfile(
        tmp_path / "checkpoints/metadata.pkl"
    ), f"No file found at {tmp_path / 'checkpoints/metadata.pkl'}"

    # add check checkpoint integrity callback for second run
    train_args["create_checkpoint_integrity_callback"] = True
    train_args["checkpoint_integrity_callback_kwargs.metadata_path"] = tmp_path
    # remove kill after checkpoint and metadata save callbacks for second run
    train_args["create_kill_after_signal_callback"] = False
    train_args["kill_after_signal_callback_kwargs"] = None
    train_args["create_metadata_save_callback"] = False
    train_args["metadata_save_callback_kwargs.metadata_path"] = None

    cmd = f"python {script_path} ++exp_manager.exp_dir={tmp_path} {data_str} " + " ".join(
        f"++{k}={v}" for k, v in train_args.items()
    )
    cmd = cmd + f' "++checkpoint_integrity_callback_kwargs.metadata_keys={metadata_keys}"'
    # TODO(dorotat) Trye to simplify  when data-related utils for ESM2 are refactored
    if "esm2" not in script_path and "dnabert" not in script_path:
        cmd += " " + " ".join(f"++{k}={v}" for k, v in data_args.items())
    print(cmd)
    # run resume training run to load checkpoint and check against saved metadata
    process_handle = subprocess.run(cmd, shell=True, capture_output=True)
    error_out = process_handle.stderr.decode("utf-8")
    assert process_handle.returncode == 0, f"Resuming training command failed:\n{cmd}\n Error log:\n{error_out}"

'''