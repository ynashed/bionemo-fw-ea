# Download nemo checkpoint
python download_artifacts.py --models esm2nv_650m --model_dir models/protein/esm2nv/

# Prepare FLIP data
python examples/protein/downstream/downstream_flip.py \
    --config-path=/workspace/bionemo/examples/protein/esm2nv/conf \
    --config-name=downstream_flip_sec_str \
    ++do_preprocessing=True \
    ++do_training=False \
    ++do_testing=False

# FLIP downstream training
python examples/protein/downstream/downstream_flip.py \
    --config-path=/workspace/bionemo/examples/protein/esm2nv/conf \
    --config-name=downstream_flip_sec_str \
    ++trainer.max_epochs=2 \
    ++trainer.val_check_interval=1 \
    ++trainer.devices=8 \
    ++model.accumulate_grad_batches=64

# checkpoint resumption
python examples/protein/downstream/downstream_flip.py \
    --config-path=/workspace/bionemo/examples/protein/esm2nv/conf \
    --config-name=downstream_flip_sec_str \
    ++trainer.max_epochs=5 \
    ++trainer.val_check_interval=1 \
    ++trainer.devices=8 \
    ++model.accumulate_grad_batches=64 \
    ++restore_from_path=results/nemo_experiments/esm2nv_flip/esm2nv_flip_secondary_structure_finetuning_encoder_frozen_True/checkpoints/esm2nv_flip.nemo \
    ++exp_manager.wandb_logger_kwargs.name=esm2nv_flip_secondary_structure_finetuning_encoder_frozen_True_restore
