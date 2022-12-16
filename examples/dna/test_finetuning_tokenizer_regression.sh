# This file tests the ability to properly restore a DNABERT model during
# finetuning from a .nemo file. The items that had previously prevented this
# were 1) a dependence on the training data locations in the intialization
# of a DNABERT model, such that if those paths did not exist on the system
# loading the `.nemo` file, the model would not be loaded properly. 2) A
# similar issue existed for the tokenizers, however as the tokenizers are
# essential to the model, they can be registered with the `.nemo` file.
#
# This test exercises both of these abilities, ensuring that neither the
# training data, nor the original tokenizers have to exist in order for the
# .nemo file to be loaded.
set +x
MODEL=/tmp/dnabert_tokenizer.model
VOCAB=/tmp/dnabert_tokenizer.vocab
EXPDIR=/tmp/dnabert_finetuning_tokenizer_test
NEMO_FILE=$EXPDIR/checkpoints/dnabert.nemo
DATASET_DIR=/tmp/dnabert_finetune_test_pretrain
FINETUNE_DIR=/tmp/dnabert_finetune_test_finetune
rm -rf $EXPDIR

rm -f $MODEL
rm -f $VOCAB
rm -rf $DATASET_DIR

set -e
python dnabert_pretrain.py ++do_preprocess=True \
    ++trainer.devices=2 \
    ++trainer.max_steps=4 \
    ++trainer.val_check_interval=2 \
    ++exp_manager.checkpoint_callback_params.always_save_nemo=True \
    ++exp_manager.explicit_log_dir=$EXPDIR \
    ++model.data.dataset_path=$DATASET_DIR \
    ++model.data.discretize_train=True \
    ++model.data.dataset.train="chr1.fna.gz.chunked.fa" \
    ++model.tokenizer.k=3 \
    ++model.tokenizer.model=$MODEL \
    ++model.tokenizer.vocab_file=$VOCAB

rm -f $MODEL
rm -f $VOCAB
rm -r $DATASET_DIR
rm -rf $FINETUNE_DIR

python splice_site_finetune.py ++task.do_preprocess=True \
    ++task.do_training=True \
    ++task.trainer.devices=2 \
    ++task.trainer.max_steps=4 \
    ++task.trainer.val_check_interval=2 \
    ++task.model.data.dataset_path=$FINETUNE_DIR \
    ++task.model.data.train_file=$FINETUNE_DIR/splits/train.csv \
    ++task.model.encoder.checkpoint=$NEMO_FILE
