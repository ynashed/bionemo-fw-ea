# Introduction

This README guides you through the process of converting a Hugging Face (HF) model checkpoint into a NVIDIA NeMo checkpoint and performing inference with different Tensor Model Parallelism (TP) configurations. The script `run-full-comparison.sh` includes all the necessary steps, providing a modular and easy-to-follow structure.

# Prerequisites
- **ESM META Code Installation**: Ensure you have the ESM META code installed to convert META checkpoints to HuggingFace formats. Run:
```bash
pip install "fair-esm[esmfold]"
```
*Note*: If you already have downloaded the HF ESM2 checkpoints, you can skip this step.

- **Tensor Model Parallelism Conversio**n: If you're only interseted in the TP conversion of a converted ESM2nv model, you can download the ESM2nv NeMo checkpoints using the download command. Then, follow the instructions of step `3.` in the [Step by Step instructions](#3-change-the-tensor-parallel-partition-of-esm2-nemo-checkpoints) section.
```bash
cd /workspace/bionemo
./launch.sh download
```


# Step by Step instructions:

## 1. Convert META ESM2 checkpoints to HF checkpoints
The first step consists of downloading ESM2 checkpoints from Meta repository and convert them to HF checkpoints:

```bash
python convert_meta_esm2_to_hf_copied.py --pytorch_dump_folder_path /data/esm_hf_checkpoints --model esm2_t33_650M_UR50D
```
More information can be found in: [Hugging Face Transformers GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/convert_esm.py)

## 2. Convert HF ESM2 checkpoints to NeMo ones
Once you have saved ESM2 HF checkpoints, you can run the script  `convert_hf_esm2_to_nemo.py` to generate ESM2 NeMo checkpoints:
```bash
python convert_hf_esm2_to_nemo.py --in-file "$input_file" --out-file "$output_file" --use_nemo_apex --precision "$precision" --run_sanity_check
```
- `--in-file`: path to the HF checkpoints
- `--out-file`: output path for the converted NeMo checkpoints.
- `--precision`: precision to use for saving the nemo weights.
- `--model_size`: ESM2 model size, supported values are "8M", "650M", "3B" and "15B"
- `--use_nemo_apex`: whether to use the optimized apex layers: FusedLayerNorm and RowParallelLinear
- `--run_sanity_check`: compare the max-absolute and relative differences between the outputs of HF ESM2 and the converted ESM2nv BioNeMo models.


## 3. Change the Tensor Parallel Partition of ESM2 NeMo checkpoints
This step assumes that you already have converted ESM2 NeMo checkpoints with TP=1. To load the weights with a different TP partition, you should run the following command:
```bash
python megatron_change_num_partitions.py --model_file=PATH_NEMO_CHECKPOINT\
--target_file=OUTPUT_PATH_NEMO_CHECKPOINT_WITH_NEW_TP_PARTITION \
--tensor_model_parallel_size=1 \
--target_tensor_model_parallel_size=$TARGET_TP_SIZE \
--pipeline_model_parallel_size=1  \
--target_pipeline_model_parallel_size=1 \
--precision=$PRECISION_OF_ORIGINAL_WEIGHTS \
--model_class "bionemo.model.protein.esm1nv.base.ESMnvMegatronBertModel" \
--tp_conversion_only \
--tokenizer_model_name="facebook/esm2_t33_650M_UR50D"
```

*Note*: The script supports TP partition change only for BERT-like models. PP partitioning is not supported.

## 4. Run inference with different TP partitions
```bash
bash infer.sh --config-name "INFERENCE_CONF_TO_USE" ++model.downstream_task.restore_from_path="PATH_TO_NEMO_CHECKPOINTS" ++model.data.output_fname="OUTPUT_PATH_TO_PKL_FILE"
```

*Note:* Note: Ensure the trainer.devices in the inference config file matches the TP size (model.tensor_model_parallel_size).

# Full Conversion Script

The run-full-conversion.sh script automates the process of converting a Hugging Face (HF) model checkpoint to a NVIDIA NeMo checkpoint and performing inference with different Tensor Model Parallelism (TP) configurations. The script is designed to be flexible, allowing users to skip certain steps if needed. Below is an explanation of the script's structure and functionalities.
## Script Structure

### Variables Definition
- ESM_MODEL: Specifies the model to be used, with options like "esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D", etc.
- TARGET_TP_SIZE: The target tensor parallelism size.
- PRECISION: Precision level ("16" for 16-bit precision).
- Skip Flags: Variables like SKIP_META_HF_CONVERSION to control whether to skip specific conversion steps.
- File Paths: Paths for the checkpoint files and folder locations.
- Converters: Specifies the scripts used for conversion (NEMO_CONVERTER and HF_CONVERTER).

### Functions Definition
- print_step: Displays the current step being executed.
convert_to_hf_checkpoint: Converts META checkpoints to HF format.
- convert_to_nemo_checkpoint: Converts HF checkpoints to NeMo format.
- change_num_partitions: Adjusts the number of TP partitions in a NeMo model.
- perform_inference: Runs inference using a specific configuration and model.

## Usage

```bash
./run-full-conversion.sh
```

**Note** you can modify the variables at the top to suit your requirements. For instance, you can choose a different ESM model or precision level. Also, if you wish to skip certain conversion steps, set the corresponding skip flags to true.
