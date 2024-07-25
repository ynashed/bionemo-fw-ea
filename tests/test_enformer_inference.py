# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import subprocess
from pathlib import Path
from typing import Iterator, Literal

import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bionemo.data.dna.enformer.genome_interval import FastaDataset
from bionemo.model.dna.enformer import Enformer
from bionemo.model.utils import setup_trainer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import Deterministic, distributed_model_parallel_state


UPDATE_GOLDEN_VALUES: bool = os.environ.get("UPDATE_GOLDEN_VALUES", "0") == "1"


def _load_test_data(cfg: DictConfig, data_path: str) -> DataLoader:
    dataset = FastaDataset(fasta_file=data_path, context_length=cfg.model.context_length)
    return DataLoader(dataset, batch_size=cfg.model.micro_batch_size, num_workers=cfg.model.data.num_workers)


def _load_model(cfg: DictConfig) -> Iterator[Enformer]:
    with Deterministic(), distributed_model_parallel_state():
        trainer = setup_trainer(cfg)
        enformer = Enformer.restore_from(restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer)
        yield enformer


def _update_infer_cfg(config_path: str, restore_from_path: str) -> DictConfig:
    cfg = load_model_config(config_name="enformer_infer", config_path=config_path)
    cfg.trainer.devices = 1
    cfg.exp_manager.exp_dir = None
    cfg.restore_from_path = restore_from_path
    cfg.model.predictions_output_path = None
    return cfg


@pytest.fixture(scope="module")
def config_path(bionemo_home: Path) -> str:
    path = bionemo_home / "examples" / "dna" / "enformer" / "conf"
    return str(path)


@pytest.fixture(scope="module")
def data_path(bionemo_home: Path) -> str:
    path = bionemo_home / "examples" / "tests" / "test_data" / "dna" / "test" / "chr1-test.fa"
    return str(path)


@pytest.fixture(scope="module")
def enformer_inferer_32(bionemo_home: Path, config_path: str) -> Enformer:
    enformer_pretrained32_path = bionemo_home / "models" / "dna" / "enformer" / "enformer_finetuned32.nemo"
    cfg = _update_infer_cfg(config_path, enformer_pretrained32_path)
    yield from _load_model(cfg)


@pytest.fixture(scope="module")
def enformer_inferer_16(bionemo_home: Path, config_path: str) -> Enformer:
    enformer_pretrained16_path = bionemo_home / "models" / "dna" / "enformer" / "enformer_finetuned16.nemo"
    cfg = _update_infer_cfg(config_path, enformer_pretrained16_path)
    yield from _load_model(cfg)


@pytest.fixture(scope="module")
def enformer_test_data(config_path: str, data_path: str) -> DataLoader:
    cfg = load_model_config(config_name="enformer_infer", config_path=config_path)
    data_loader = _load_test_data(cfg, data_path)
    return data_loader


def detect_gpu_architecture() -> Literal["a6000", "a100", "a10g"]:
    gpu_name = torch.cuda.get_device_name(0)
    if "A6000" in gpu_name:
        return "a6000"
    elif "A100" in gpu_name:
        return "a100"
    elif "A10G" in gpu_name:
        return "a10g"
    else:
        raise ValueError(f"Unrecognized GPU architecture: {gpu_name}")


def gv_path_and_pbss_download(
    base_dir: Path,
    precision: Literal[16, 32],
    architecture: Literal["a6000", "a100", "a10g"],
    pbss_bucket: str,
    pbss_key_prefix: str,
) -> Path:
    gv_filename = f"inference_test_golden_values_fp{precision}_{architecture}.pt"

    gv_file = base_dir / gv_filename

    if not gv_file.exists():
        print("GV file not found locally, downloading from PBSS")
        gv_file.parent.mkdir(parents=True, exist_ok=True)
        ret_code = subprocess.check_call(
            [
                "aws",
                "s3",
                "cp",
                f"s3://{pbss_bucket}/{pbss_key_prefix}/{gv_filename}",
                str(gv_file),
                "--endpoint-url",
                "https://pbss.s8k.io",
            ]
        )
        if ret_code != 0:
            raise ValueError("PBSS download failed! Check logs for details.")

    assert gv_file.is_file()

    return gv_file


@pytest.fixture(scope="module")
def enformer_expected_vals_fp32_path(bionemo_home: Path) -> Path:
    return gv_path_and_pbss_download(
        base_dir=bionemo_home / "tests" / "data" / "enformer",
        precision=32,
        architecture=detect_gpu_architecture(),
        pbss_bucket="bionemo-ci",
        pbss_key_prefix="test-data/enformer",
    )


@pytest.fixture(scope="module")
def enformer_expected_vals_fp16_path(bionemo_home: Path) -> Path:
    return gv_path_and_pbss_download(
        base_dir=bionemo_home / "tests" / "data" / "enformer",
        precision=16,
        architecture=detect_gpu_architecture(),
        pbss_bucket="bionemo-ci",
        pbss_key_prefix="test-data/enformer",
    )


def _test_enformer_inference(
    enformer_inferer: Enformer, enformer_test_data: DataLoader, enformer_expected_vals_path: Path
) -> None:
    if UPDATE_GOLDEN_VALUES:
        os.makedirs(os.path.dirname(enformer_expected_vals_path), exist_ok=True)
    else:
        assert os.path.exists(
            enformer_expected_vals_path
        ), f"Expected values file not found at {enformer_expected_vals_path}. Rerun with UPDATE_GOLDEN_VALUES=1 to create it."

    trainer = enformer_inferer.trainer
    predictions = trainer.predict(enformer_inferer, enformer_test_data)
    expression_preds = predictions[0]["pred"]

    expression_preds2 = trainer.predict(enformer_inferer, enformer_test_data)[0]["pred"]
    expression_preds3 = trainer.predict(enformer_inferer, enformer_test_data)[0]["pred"]

    assert enformer_inferer.training is False
    assert expression_preds is not None
    assert len(expression_preds.shape) == 3

    torch.testing.assert_close(expression_preds3, expression_preds2, rtol=None, atol=None, equal_nan=True)
    torch.testing.assert_close(expression_preds, expression_preds2, rtol=None, atol=None, equal_nan=True)

    if UPDATE_GOLDEN_VALUES:
        torch.save(
            {
                "expected_predictions": expression_preds,
            },
            str(enformer_expected_vals_path),
        )
        assert False, f"Updated expected values at {enformer_expected_vals_path}, rerun with UPDATE_GOLDEN_VALUES=0"

    else:
        expected_vals = {k: v.to(expression_preds.device) for k, v in torch.load(enformer_expected_vals_path).items()}
        assert torch.allclose(
            expression_preds, expected_vals["expected_predictions"], atol=1e-6
        ), "Difference detected!"


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_enformer_inference_fp32(
    enformer_inferer_32: Enformer, enformer_test_data: DataLoader, enformer_expected_vals_fp32_path: Path
) -> None:
    _test_enformer_inference(enformer_inferer_32, enformer_test_data, enformer_expected_vals_fp32_path)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_enformer_inference_fp16(
    enformer_inferer_16: Enformer, enformer_test_data: DataLoader, enformer_expected_vals_fp16_path: Path
) -> None:
    _test_enformer_inference(enformer_inferer_16, enformer_test_data, enformer_expected_vals_fp16_path)
