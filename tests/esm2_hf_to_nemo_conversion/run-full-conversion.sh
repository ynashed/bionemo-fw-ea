#!/bin/bash
set -e
set -x
########################
# Variables definition #
########################
SKIP_META_HF_CONVERSION=false # Convert META to HF Checkpoint
SKIP_HF_NEMO_CONVERSION=false # Convert HF to NeMo Checkpoint
SKIP_TP_CONVERSION=false # Change TP Partitions of NeMo checkpoint
CHECK_TP_CONVERSION=true # Performs inference and compares results 

ESM_MODEL="esm2_t6_8M_UR50D" # possible values: "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"
TARGET_TP_SIZE=2 # The target size for Tensor Parallel for changing the model's partitions (if `SKIP_TP_CONVERSION` is false)
PRECISION="16" # The float precision od the converted checkpoint

HF_DUMP_FOLDER="/data/esm_hf_checkpoints"
NEMO_DUMP_FOLDER="/data/esm2_nemo_checkpoints"
HF_CKPT="${HF_DUMP_FOLDER}${ESM_MODEL}"
TP1_NEMO_FILE="${NEMO_DUMP_FOLDER}/converted_${ESM_MODEL}.nemo"
TP2_NEMO_FILE="${NEMO_DUMP_FOLDER}/converted_${ESM_MODEL}_TP-${TARGET_TP_SIZE}.nemo"

WORKTREE=${WORKTREE:-"/workspace/bionemo"}
NEMO_CONVERTER=${NEMO_CONVERTER:-"convert_hf_esm2_to_nemo.py"}
HF_CONVERTER=${HF_CONVERTER:-"convert_meta_esm2_to_hf_copied.py"}


########################
# Functions definition #
########################

# Function to print step messages
print_step() {
    local message="$1"
    echo -e "\nStep: $message \n"
}

# Function to convert META checkpoint to HF checkpoint
convert_to_hf_checkpoint() {
    local hf_dump_folder="$1"
    local model="$2"
    print_step "Converting META checkpoint to HF checkpoint"
    python "$HF_CONVERTER" --pytorch_dump_folder_path "$hf_dump_folder" --model "$model"
}

# Function to convert HF checkpoint to NEMO checkpoint
convert_to_nemo_checkpoint() {
    local input_file="$1"
    local output_file="$2"
    local precision="$3"
    print_step "Converting HF checkpoint to NEMO checkpoint"
    python "$NEMO_CONVERTER" --in-file "$input_file" --out-file "$output_file" --use_nemo_apex --precision "$precision" --run_sanity_check
}

# Function to change the number of partitions
change_num_partitions() {
    local model_file="$1"
    local target_file="$2"
    local target_tp_partition="$3"

    print_step "Changing the number of partitions to TP = $target_tp_partition"

    python megatron_change_num_partitions.py --model_file="$model_file" \
        --target_file="$target_file" \
        --tensor_model_parallel_size=1 \
        --target_tensor_model_parallel_size="$target_tp_partition" \
        --pipeline_model_parallel_size=1 \
        --target_pipeline_model_parallel_size=1 \
        --precision=32 \
        --model_class "bionemo.model.protein.esm1nv.base.ESMnvMegatronBertModel" \
        --tp_conversion_only \
        --tokenizer_model_name="facebook/esm2_t33_650M_UR50D"
}

# Function to perform inference
perform_inference() {
    local config_name="$1"
    local restore_from_path="$2"
    filename=$(basename "$restore_from_path")
    print_step "Performing inference for $filename"
    bash infer.sh --config-name "$config_name" ++model.downstream_task.restore_from_path="$restore_from_path" ++model.data.output_fname="/data/${filename}"
}

########################
#   Execute commands   #
########################

if [ "$SKIP_META_HF_CONVERSION" = false ]; then
    # Convert META checkpoint to HF checkpoint
    convert_to_hf_checkpoint "$HF_CKPT" "$ESM_MODEL"
else
    echo "Skipping META to HF checkpoint conversion."
fi

# Convert HF checkpoint to NEMO checkpoint
if [ "$SKIP_HF_NEMO_CONVERSION" = false ]; then
    rm -f "$TP1_NEMO_FILE" 
    convert_to_nemo_checkpoint "$HF_CKPT" "$TP1_NEMO_FILE" "$PRECISION"
else
    echo "Skipping HF to NEMO checkpoint conversion."
fi

# Change the number of partitions
if [ "$SKIP_TP_CONVERSION" = false ]; then
    rm -f "$TP2_NEMO_FILE"
    change_num_partitions "$TP1_NEMO_FILE" "$TP2_NEMO_FILE" "$TARGET_TP_SIZE"
    if [ "$CHECK_TP_CONVERSION" = true ]; then
        # Perform inference for TP1 and TP2
        cd "$WORKTREE/examples/protein/esm2nv"
        perform_inference "infer_esm2_tp1" "$TP1_NEMO_FILE"
        perform_inference "infer_esm2_tp2" "$TP2_NEMO_FILE"

        # Compare results
        print_step "Comparing results of converted ESM2 model with TP=1 and TP=${TARGET_TP_SIZE} partitions"
        cd "$WORKTREE/internal/esm2_hf_to_nemo_conversion/"
        pkl_tp1=$(basename "$TP1_NEMO_FILE")
        pkl_tp2=$(basename "$TP2_NEMO_FILE")
        python compare.py --prediction_file_1 "/data/${pkl_tp1}" --prediction_file_2 "/data/${pkl_tp2}"
    fi
else
    echo "Skipping TP partition conversion."
fi 
